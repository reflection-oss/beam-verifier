// The sandbox connector forwards model requests to the host-side proxy through
// an authenticated SSH connection over WebSocket. Agents call its sandbox port.
//
// The connector needs no Go, Python, or system C libraries in the sandbox.
// Its launcher needs bash.
// Usage: tunnel-connector CONFIG_PATH. The private JSON config is read and deleted
// before accepting connections through the authenticated Daytona preview.
package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"golang.org/x/crypto/ssh"
)

const (
	sshUser      = "beam-verifier"
	loginTimeout = 10 * time.Second
	// The host client receives frames of at most 64KiB; chunk writes below that.
	frameLimit = 65536
	writeChunk = 32768
	// Evict a silent peer so its model listener frees for reconnection, matching
	// the host client's 20s ping interval and 20s pong timeout.
	pingInterval = 20 * time.Second
	pongWait     = 40 * time.Second
)

type config struct {
	HostKey       string `json:"host_key"`
	Password      string `json:"password"`
	ModelPort     int    `json:"model_port"`
	WebsocketPort int    `json:"websocket_port"`
}

// wsConn lets the SSH server read and write bytes through a WebSocket.
type wsConn struct {
	ws      *websocket.Conn
	reader  io.Reader
	writeMu sync.Mutex
}

func (c *wsConn) Read(p []byte) (int, error) {
	for {
		if c.reader == nil {
			kind, reader, err := c.ws.NextReader()
			if err != nil {
				return 0, err
			}
			if kind != websocket.BinaryMessage {
				return 0, errors.New("SSH transport requires binary WebSocket frames")
			}
			c.reader = reader
		}
		n, err := c.reader.Read(p)
		if err == io.EOF {
			c.reader = nil
			if n == 0 {
				continue
			}
			err = nil
		}
		return n, err
	}
}

func (c *wsConn) Write(p []byte) (int, error) {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	for sent := 0; sent < len(p); {
		chunk := min(len(p)-sent, writeChunk)
		if err := c.ws.WriteMessage(websocket.BinaryMessage, p[sent:sent+chunk]); err != nil {
			return sent, err
		}
		sent += chunk
	}
	return len(p), nil
}

func (c *wsConn) Close() error                       { return c.ws.Close() }
func (c *wsConn) LocalAddr() net.Addr                { return c.ws.LocalAddr() }
func (c *wsConn) RemoteAddr() net.Addr               { return c.ws.RemoteAddr() }
func (c *wsConn) SetReadDeadline(t time.Time) error  { return c.ws.SetReadDeadline(t) }
func (c *wsConn) SetWriteDeadline(t time.Time) error { return c.ws.SetWriteDeadline(t) }

func (c *wsConn) SetDeadline(t time.Time) error {
	if err := c.ws.SetReadDeadline(t); err != nil {
		return err
	}
	return c.ws.SetWriteDeadline(t)
}

type forwardRequest struct {
	Addr string
	Port uint32
}

type forwardedAddress struct {
	DestAddr string
	DestPort uint32
	OrigAddr string
	OrigPort uint32
}

// serveSSH authenticates the host and forwards model requests until it disconnects.
// It then closes the sandbox port so a new connection can use it.
func serveSSH(conn net.Conn, serverConfig *ssh.ServerConfig, modelPort int) {
	defer func() { _ = conn.Close() }()
	timer := time.AfterFunc(loginTimeout, func() { _ = conn.Close() })
	sshConn, channels, requests, err := ssh.NewServerConn(conn, serverConfig)
	timer.Stop()
	if err != nil {
		return
	}
	defer func() { _ = sshConn.Close() }()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		for channel := range channels {
			_ = channel.Reject(ssh.Prohibited, "this tunnel only serves remote forwarding")
		}
	}()
	var listener net.Listener
	defer func() {
		if listener != nil {
			_ = listener.Close()
		}
	}()
	for request := range requests {
		switch request.Type {
		case "tcpip-forward":
			var forward forwardRequest
			if ssh.Unmarshal(request.Payload, &forward) != nil ||
				forward.Addr != "127.0.0.1" || forward.Port != uint32(modelPort) || listener != nil {
				_ = request.Reply(false, nil)
				continue
			}
			bound, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(modelPort)))
			if err != nil {
				_ = request.Reply(false, nil)
				continue
			}
			listener = bound
			// Reply before accepting, so no forwarded-tcpip open can precede it.
			_ = request.Reply(true, nil)
			go forwardConnections(ctx, sshConn, bound, uint32(modelPort))
		case "cancel-tcpip-forward":
			if listener != nil {
				_ = listener.Close()
				listener = nil
			}
			_ = request.Reply(true, nil)
		default:
			_ = request.Reply(false, nil)
		}
	}
}

func forwardConnections(ctx context.Context, sshConn *ssh.ServerConn, listener net.Listener, modelPort uint32) {
	for {
		tcp, err := listener.Accept()
		if err != nil {
			return
		}
		go relay(ctx, sshConn, tcp, modelPort)
	}
}

func relay(ctx context.Context, sshConn *ssh.ServerConn, tcp net.Conn, modelPort uint32) {
	// Tunnel loss must interrupt TCP reads even if the sandbox client stays open.
	stopClose := context.AfterFunc(ctx, func() { _ = tcp.Close() })
	defer stopClose()
	defer func() { _ = tcp.Close() }()
	origin, ok := tcp.RemoteAddr().(*net.TCPAddr)
	if !ok {
		return
	}
	payload := forwardedAddress{
		DestAddr: "127.0.0.1",
		DestPort: modelPort,
		OrigAddr: origin.IP.String(),
		OrigPort: uint32(origin.Port),
	}
	channel, requests, err := sshConn.OpenChannel("forwarded-tcpip", ssh.Marshal(&payload))
	if err != nil {
		return
	}
	defer func() { _ = channel.Close() }()
	go ssh.DiscardRequests(requests)
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = io.Copy(channel, tcp)
		_ = channel.CloseWrite()
	}()
	_, _ = io.Copy(tcp, channel)
	if half, ok := tcp.(*net.TCPConn); ok {
		_ = half.CloseWrite()
	}
	<-done
}

func main() {
	log.SetFlags(0)
	if len(os.Args) != 2 {
		log.Fatalf("usage: %s CONFIG_PATH", os.Args[0])
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	// Consume the ephemeral credentials before accepting any connection.
	if err := os.Remove(os.Args[1]); err != nil {
		log.Fatal(err)
	}
	var tunnelConfig config
	if err := json.Unmarshal(raw, &tunnelConfig); err != nil {
		log.Fatal("the tunnel config is not valid JSON")
	}
	if tunnelConfig.Password == "" || tunnelConfig.ModelPort == 0 || tunnelConfig.WebsocketPort == 0 {
		log.Fatal("the tunnel config is missing a credential or port")
	}
	signer, err := ssh.ParsePrivateKey([]byte(tunnelConfig.HostKey))
	if err != nil {
		log.Fatal("the tunnel config holds no parseable SSH host key")
	}
	serverConfig := &ssh.ServerConfig{
		PasswordCallback: func(meta ssh.ConnMetadata, password []byte) (*ssh.Permissions, error) {
			userOK := subtle.ConstantTimeCompare([]byte(meta.User()), []byte(sshUser))
			passwordOK := subtle.ConstantTimeCompare(password, []byte(tunnelConfig.Password))
			if userOK&passwordOK == 1 {
				return nil, nil
			}
			return nil, errors.New("authentication failed")
		},
	}
	serverConfig.AddHostKey(signer)
	listener, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", tunnelConfig.WebsocketPort))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("SSH WebSocket connector ready")
	upgrader := websocket.Upgrader{
		ReadBufferSize:  frameLimit,
		WriteBufferSize: frameLimit,
		CheckOrigin:     func(*http.Request) bool { return true },
	}
	log.Fatal(http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ws, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		ws.SetReadLimit(frameLimit)
		alive := func(string) error { return ws.SetReadDeadline(time.Now().Add(pongWait)) }
		_ = alive("")
		ws.SetPongHandler(alive)
		ws.SetPingHandler(func(payload string) error {
			_ = ws.WriteControl(websocket.PongMessage, []byte(payload), time.Now().Add(10*time.Second))
			return alive("")
		})
		stopPings := make(chan struct{})
		defer close(stopPings)
		go func() {
			ticker := time.NewTicker(pingInterval)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					if ws.WriteControl(websocket.PingMessage, nil, time.Now().Add(10*time.Second)) != nil {
						return
					}
				case <-stopPings:
					return
				}
			}
		}()
		serveSSH(&wsConn{ws: ws}, serverConfig, tunnelConfig.ModelPort)
	})))
}

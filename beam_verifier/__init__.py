"""beam-verifier — a basic eval library for OpenAI-compatible endpoints.

Endpoint workflows:
  * eval    — reproduce benchmark scores (GPQA-Diamond, AIME) and print them
  * verify  — check endpoint behavior with the functional conformance tests
              (params / request validation / chat / tool-calling / ...)
"""

# Load .env once so endpoint credentials are available without manual exports.
try:
    import dotenv

    dotenv.load_dotenv()
except Exception:
    pass

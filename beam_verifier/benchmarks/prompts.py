"""Shared class for Beam prompts."""

import dataclasses
import string


@dataclasses.dataclass
class BeamPromptFactory:
    """Beam prompt text.

    Attributes:
      name: Versioned identity recorded in the native harness artifacts.
      user_template: Optional template with `prompt` and `workdir` placeholders.
      system_text: Optional benchmark instructions for the system message.
    """

    name: str
    user_template: str | None = None
    system_text: str | None = None

    @property
    def requires_workdir(self) -> bool:
        """Whether rendering needs the runner's sandbox working directory."""
        return any(field == "workdir" for _, field, _, _ in string.Formatter().parse(self.user_template or ""))

    def render_task_instruction(self, prompt: str, *, workdir: str | None = None) -> str:
        """Renders task text, preserving placeholders inside the input.

        Args:
          prompt: Original task text; returned unchanged without a template.
          workdir: Working directory resolved by the runner in its sandbox.

        Returns:
          The user instruction to send to the model.

        Raises:
          ValueError: If the template requires a missing working directory.
        """
        if self.user_template is None:
            return prompt
        if self.requires_workdir and workdir is None:
            raise ValueError(f"Prompt factory {self.name!r} requires a sandbox working directory")
        return self.user_template.format(prompt=prompt, workdir=workdir)

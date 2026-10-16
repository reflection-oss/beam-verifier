"""Shared class for Beam prompts."""

import dataclasses


@dataclasses.dataclass
class BeamPromptFactory:
    """Beam prompt text.

    Attributes:
      name: Versioned identity recorded in the native harness artifacts.
      user_template: Optional template with a `prompt` placeholder.
      system_text: Optional benchmark instructions for the system message.
    """

    name: str
    user_template: str | None = None
    system_text: str | None = None

    def render_user(self, prompt: str) -> str:
        """Renders task text, preserving placeholders inside the input.

        Args:
          prompt: Original task text; returned unchanged without a template.

        Returns:
          The user instruction to send to the model.
        """
        if self.user_template is None:
            return prompt
        return self.user_template.format(prompt=prompt)

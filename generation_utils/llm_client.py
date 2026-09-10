import os, json, re
from typing import Type, Any
from dotenv import load_dotenv
from pydantic import BaseModel
import streamlit as st
from generation_utils.schema import Response

# --- IMPORTS FOR PROVIDERS ---
try:
    from google import genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from groq import Groq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


class LLMClient:
    def __init__(self, provider: str, model_name: str):
        load_dotenv()
        self.provider = provider.lower()
        self.model_name = model_name
        self._init_client()
        print(f"🔌 LLM Client connected: {self.provider}/{self.model_name}")

    def _init_client(self):
        # Helper to fetch key from env first, then streamlit secrets if available
        def get_key(env_name, secret_path):
            val = os.getenv(env_name)
            if val:
                return val
            try:
                if hasattr(st, "secrets") and st.secrets:
                    return st.secrets.get("gkeys", {}).get(secret_path) or st.secrets.get(env_name)
            except Exception:
                pass
            return None

        if self.provider == "gemini":
            if not HAS_GEMINI: raise ImportError("Run `pip install google-genai`")
            api_key = get_key("GEMINI_API_KEY", "gemini")
            self.client = genai.Client(api_key=api_key)

        elif self.provider == "openai":
            if not HAS_OPENAI: raise ImportError("Run `pip install openai`")
            api_key = get_key("OPENAI_API_KEY", "openai")
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            if not HAS_ANTHROPIC: raise ImportError("Run `pip install anthropic`")
            api_key = get_key("ANTHROPIC_API_KEY", "anthropic")
            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "groq":
            if not HAS_GROQ: raise ImportError("Run `pip install groq`")
            api_key = get_key("GROQ_API_KEY", "groq")
            self.client = Groq(api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate_text(self, prompt: str, system_instruction: str = None) -> str:
        """Standard text generation for the Student."""
        try:
            # --- GEMINI ---
            if self.provider == "gemini":
                full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config={"temperature": 0.0}
                ).text

            # --- OPENAI ---
            elif self.provider == "openai":
                messages = [{"role": "user", "content": prompt}]
                if system_instruction:
                    messages.insert(0, {"role": "system", "content": system_instruction})
                res = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=0.0, seed=42
                )
                return res.choices[0].message.content

            # --- ANTHROPIC ---
            elif self.provider == "anthropic":
                messages = [{"role": "user", "content": prompt}]
                if system_instruction:
                    prompt = f"{system_instruction}\n\n{prompt}"
                res = self.client.messages.create(
                    model=self.model_name, max_tokens=1024, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )
                return res.content[0].text

            # --- GROQ ---
            elif self.provider == "groq":
                messages = [{"role": "user", "content": prompt}]
                if system_instruction:
                    messages.insert(0, {"role": "system", "content": system_instruction})
                res = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=0.0, seed=42
                )
                return res.choices[0].message.content

        except Exception as e:
            return f"Error: {e}"

    def generate_structured(self, prompt: str, schema_model: Type[BaseModel], system_instruction: str = None) -> Any:
        """Structured output """
        try:
            # --- GEMINI STRUCTURED ---
            if self.provider == "gemini":
                full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema_model,
                        "temperature": 0.0
                    }
                )
                return schema_model.model_validate_json(response.text)

            # --- OPENAI STRUCTURED ---
            elif self.provider == "openai":
                messages = [{"role": "user", "content": prompt}]
                if system_instruction:
                    messages.insert(0, {"role": "system", "content": system_instruction})
                completion = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=schema_model,
                    temperature=0.0,
                    seed=42
                )
                return completion.choices[0].message.parsed

            elif self.provider == "groq":
                # https://console.groq.com/docs/structured-outputs
                schema_json = schema_model.model_json_schema()
                sys_msg = (
                    f"{system_instruction}\n\n" if system_instruction else ""
                ) + f"You are a helpful assistant that outputs JSON matching this JSON schema:\n{json.dumps(schema_json)}"
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt}
                ]
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        seed=42,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_model.__name__,
                                "schema": schema_json
                            }
                        }
                    )
                except Exception as groq_err:
                    # Fallback to json_object if the model does not support response_format json_schema
                    if "json_schema" in str(groq_err) or "400" in str(groq_err):
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=0.0,
                            seed=42,
                            response_format={"type": "json_object"}
                        )
                    else:
                        raise groq_err

                raw_content = response.choices[0].message.content.strip()
                # Clean markdown backticks if present
                if raw_content.startswith("```"):
                    raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
                    raw_content = re.sub(r"\n?```$", "", raw_content)

                return schema_model.model_validate_json(raw_content)

            else:
                raise NotImplementedError("Structured output only supported for Gemini, Groq, and OpenAI.")

        except Exception as e:
            print(f"⚠️ Structured Generation Error: {e}")
            raise RuntimeError(f"Structured generation failed: {e}") from e

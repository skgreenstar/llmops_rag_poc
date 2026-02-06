import os
from langfuse import Langfuse
from app.core.config import get_settings
from app.core.prompts import prompt_manager

settings = get_settings()

def seed_prompts():
    print("🚀 Seeding Prompts to Langfuse...")
    langfuse = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST
    )

    defaults = prompt_manager._defaults
    
    for name, template in defaults.items():
        print(f"Creating prompt: {name}")
        try:
            # Check if exists (optional, create_prompt usually creates a new version)
            langfuse.create_prompt(
                name=name,
                prompt=template,
                config={"model": "gpt-4o", "temperature": 0},
                labels=["production"] # Important: Label as production to be picked up
            )
            print(f"✅ Created/Updated: {name}")
        except Exception as e:
            print(f"❌ Failed to create {name}: {e}")

    print("✨ Seeding Complete!")

if __name__ == "__main__":
    seed_prompts()

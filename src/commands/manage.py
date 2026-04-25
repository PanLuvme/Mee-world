"""
Slash commands for managing Mees.

v5 changes (Phase 5):
- DEFAULT_IMAGE updated to new default Mee avatar
- /mymee command group: any Discord user can create/manage their own Mee
  with their own API key (Groq free, OpenAI, OpenRouter, etc.)
- OWNER_API_KEY env: server owner can provide a fallback key so users
  don't need their own
- is_mee_owner() predicate for user-scoped commands
- validate_api_key() helper — verifies key before saving
- Groq onboarding embed shown after /mymee create for new users
- Autocomplete decorator order fixed (mee_status + mee_mood defined before registration)
- All v3/v4 admin commands retained under /mee group (owner-only)
"""
import discord
from discord import app_commands
from discord.ext import commands
import logging
import os

from src.utils import db, embeds
from src.utils.embeds import DEFAULT_IMAGE, groq_onboarding_embed
from src.utils.webhook import get_or_create_webhook
from src.agents.llm import LLMClient
from src.agents.vector_store import delete_memories_from_chroma, delete_collection

logger = logging.getLogger(__name__)

OWNER_ID          = int(os.getenv("OWNER_ID", "258778043944796161"))
MAX_MEES_PER_USER = int(os.getenv("MAX_MEES_PER_USER", "1"))
REQUIRE_ROLE_ID   = os.getenv("REQUIRE_ROLE_ID", "").strip()
OWNER_API_KEY          = os.getenv("OWNER_API_KEY", "").strip()
OWNER_API_BASE         = os.getenv("OWNER_API_BASE", "https://api.groq.com/openai/v1").strip()
OWNER_API_MODEL        = os.getenv("OWNER_API_MODEL", "llama-3.1-8b-instant").strip()

# ── Foreground (Gemini) server-level fallback ──────────────────────────────────
# If set, users don't need their own Gemini key — the bot owner provides one.
OWNER_GEMINI_API_KEY   = os.getenv("OWNER_GEMINI_API_KEY", "").strip()
OWNER_GEMINI_MODEL     = os.getenv("OWNER_GEMINI_MODEL", "gemini-2.0-flash").strip()

DEFAULT_GROQ_BASE  = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GEMINI_API_BASE    = "https://generativelanguage.googleapis.com/v1beta/openai"


# ─── Permission helpers ────────────────────────────────────────────────────────

def is_owner():
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.user.id != OWNER_ID:
            await interaction.response.send_message(
                embed=embeds.error_embed("Only the server owner can use `/mee` management commands.\n"
                                        "Use `/mymee` to manage your own Mee."),
                ephemeral=True,
            )
            return False
        return True
    return app_commands.check(predicate)


async def _check_mee_ownership(interaction: discord.Interaction, mee_name: str) -> bool:
    """Returns True if the interaction user owns the named Mee, or is the server owner."""
    if interaction.user.id == OWNER_ID:
        return True
    mee = await db.get_mee(mee_name)
    if not mee:
        return False
    return str(interaction.user.id) == str(mee.get("owner_discord_id", "0"))


async def _check_create_permission(interaction: discord.Interaction) -> bool:
    """Check REQUIRE_ROLE_ID gate and per-user Mee limit."""
    if REQUIRE_ROLE_ID:
        member = interaction.guild.get_member(interaction.user.id) if interaction.guild else None
        if not member or not any(str(r.id) == REQUIRE_ROLE_ID for r in member.roles):
            await interaction.followup.send(
                embed=embeds.error_embed("You need a specific server role to create a Mee."),
                ephemeral=True,
            )
            return False

    count = await db.count_mees_by_owner(str(interaction.user.id))
    if count >= MAX_MEES_PER_USER:
        await interaction.followup.send(
            embed=embeds.error_embed(
                f"You already have {count} Mee(s) — the limit is {MAX_MEES_PER_USER}.\n"
                "Use `/mymee delete` to remove your existing Mee first."
            ),
            ephemeral=True,
        )
        return False
    return True


async def validate_api_key(api_key: str, api_base: str, model: str) -> bool:
    """Test an API key with a minimal 1-token completion. Returns True if valid."""
    try:
        client = LLMClient(api_key=api_key, model=model, api_base=api_base)
        result = await client.complete(
            [{"role": "user", "content": "Hi"}], max_tokens=5
        )
        return bool(result)
    except Exception:
        return False


# ─── Admin modals (unchanged logic, updated DEFAULT_IMAGE) ───────────────────

class AddMeeModal(discord.ui.Modal, title="✨ Create a New Mee (Admin)"):
    name = discord.ui.TextInput(
        label="Name", placeholder="e.g. Luna", max_length=32, required=True
    )
    identity = discord.ui.TextInput(
        label="Identity (who are they?)", style=discord.TextStyle.paragraph,
        placeholder="A rich paragraph describing backstory, personality, occupation...",
        max_length=1500, required=True,
    )
    traits = discord.ui.TextInput(
        label="Traits (comma-separated)", placeholder="e.g. curious, sarcastic, warm-hearted",
        max_length=200, required=False,
    )
    goals = discord.ui.TextInput(
        label="Goals (one per line)", style=discord.TextStyle.paragraph,
        placeholder="Find true connection\nLearn something new every day",
        max_length=400, required=False,
    )
    api_config = discord.ui.TextInput(
        label="API Key | Model | Base URL (pipe-separated)",
        placeholder="gsk_xxxx|llama-3.1-8b-instant|https://api.groq.com/openai/v1",
        max_length=400, required=True,
    )

    def __init__(self, channel_id: str):
        super().__init__()
        self.channel_id = channel_id

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        parts    = [p.strip() for p in self.api_config.value.split("|")]
        api_key  = parts[0] if parts else ""
        model    = parts[1] if len(parts) > 1 else DEFAULT_GROQ_MODEL
        api_base = parts[2] if len(parts) > 2 else DEFAULT_GROQ_BASE

        traits = [t.strip() for t in self.traits.value.split(",") if t.strip()] if self.traits.value else []
        goals  = [g.strip() for g in self.goals.value.split("\n") if g.strip()] if self.goals.value else []

        if await db.get_mee(self.name.value.strip()):
            await interaction.followup.send(
                embed=embeds.error_embed(f"A Mee named **{self.name.value}** already exists."),
                ephemeral=True,
            )
            return

        channel     = interaction.guild.get_channel(int(self.channel_id))
        webhook_url = await get_or_create_webhook(channel, interaction.client.user) if channel else None

        mee_id = await db.create_mee(
            name=self.name.value.strip(), identity=self.identity.value.strip(),
            traits=traits, goals=goals, model=model, api_key=api_key, api_base=api_base,
            image_url=DEFAULT_IMAGE, channel_id=self.channel_id, webhook_url=webhook_url,
            owner_discord_id="0",   # admin-created Mees owned by server
            gemini_api_key=OWNER_GEMINI_API_KEY or None,
            gemini_model=OWNER_GEMINI_MODEL if OWNER_GEMINI_API_KEY else None,
        )

        mee   = await db.get_mee_by_id(mee_id)
        embed = embeds.mee_profile_embed(mee, memory_count=0)
        embed.title = f"🌸 {mee['name']} is born!"
        if webhook_url:
            embed.description = (embed.description or "") + "\n\n✅ Webhook auto-configured!"
        await interaction.followup.send(embed=embed, ephemeral=True)
        interaction.client.dispatch("mee_created", mee_id)


class EditIdentityModal(discord.ui.Modal, title="✏️ Edit Identity"):
    identity = discord.ui.TextInput(
        label="Identity", style=discord.TextStyle.paragraph, max_length=1500, required=True
    )

    def __init__(self, mee_data: dict):
        super().__init__()
        self.mee_data = mee_data
        self.identity.default = mee_data["identity"]

    async def on_submit(self, interaction: discord.Interaction):
        await db.update_mee(self.mee_data["id"], identity=self.identity.value.strip())
        await interaction.response.send_message(
            embed=embeds.success_embed(f"**{self.mee_data['name']}**'s identity updated!"),
            ephemeral=True,
        )


class EditModelModal(discord.ui.Modal, title="🤖 Edit LLM Settings"):
    api_config = discord.ui.TextInput(
        label="API Key | Model | Base URL",
        placeholder="gsk_xxxx|llama-3.1-8b-instant|https://api.groq.com/openai/v1",
        max_length=400, required=True,
    )

    def __init__(self, mee_data: dict):
        super().__init__()
        self.mee_data = mee_data
        self.api_config.default = (
            f"{mee_data['api_key']}|{mee_data['model']}|"
            f"{mee_data.get('api_base', DEFAULT_GROQ_BASE)}"
        )

    async def on_submit(self, interaction: discord.Interaction):
        parts    = [p.strip() for p in self.api_config.value.split("|")]
        api_key  = parts[0] if parts else self.mee_data["api_key"]
        model    = parts[1] if len(parts) > 1 else self.mee_data["model"]
        api_base = parts[2] if len(parts) > 2 else DEFAULT_GROQ_BASE

        await interaction.response.defer(ephemeral=True)
        valid = await validate_api_key(api_key, api_base, model)
        if not valid:
            await interaction.followup.send(
                embed=embeds.error_embed(
                    "❌ Could not validate this API key. Please check it and try again.\n"
                    "Make sure you're using the correct base URL for your provider."
                ),
                ephemeral=True,
            )
            return

        await db.update_mee(self.mee_data["id"], api_key=api_key, model=model, api_base=api_base)
        await interaction.followup.send(
            embed=embeds.success_embed(f"**{self.mee_data['name']}**'s LLM updated to `{model}`!"),
            ephemeral=True,
        )


class EditForegroundKeyModal(discord.ui.Modal, title="🌟 Foreground Key (Gemini)"):
    """Set the Google AI Studio API key used for direct user/Mee-to-Mee responses."""
    fg_key = discord.ui.TextInput(
        label="Google AI Studio API Key (foreground)",
        placeholder="AIza... — free at aistudio.google.com — leave blank to clear",
        max_length=400, required=False,
    )
    fg_model = discord.ui.TextInput(
        label="Gemini model (default: gemini-2.0-flash)",
        placeholder="gemini-2.0-flash",
        max_length=100, required=False,
    )

    def __init__(self, mee_data: dict):
        super().__init__()
        self.mee_data             = mee_data
        self.fg_key.default       = mee_data.get("gemini_api_key") or ""
        self.fg_model.default     = mee_data.get("gemini_model") or "gemini-2.0-flash"

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        key   = self.fg_key.value.strip()
        model = self.fg_model.value.strip() or "gemini-2.0-flash"

        if key:
            # Validate the Gemini key against the OpenAI-compat endpoint
            valid = await validate_api_key(key, GEMINI_API_BASE, model)
            if not valid:
                await interaction.followup.send(
                    embed=embeds.error_embed(
                        "❌ Could not validate the Gemini API key.\n\n"
                        "• Get a **free** key at [aistudio.google.com](https://aistudio.google.com)\n"
                        "• The key should start with `AIza`\n"
                        "• Make sure the Gemini API is enabled in your Google account"
                    ),
                    ephemeral=True,
                )
                return
            await db.update_mee(self.mee_data["id"], gemini_api_key=key, gemini_model=model)
            await interaction.followup.send(
                embed=embeds.success_embed(
                    f"✅ **{self.mee_data['name']}** will now use Gemini `{model}` "
                    f"for direct conversations with users and other Mees.\n"
                    f"Background/idle chat still uses the Groq key."
                ),
                ephemeral=True,
            )
        else:
            # Clear the foreground key — fall back to Groq for everything
            await db.update_mee(self.mee_data["id"], gemini_api_key=None, gemini_model=None)
            await interaction.followup.send(
                embed=embeds.success_embed(
                    f"Foreground key cleared. **{self.mee_data['name']}** "
                    f"will use the background (Groq) key for all calls."
                ),
                ephemeral=True,
            )


class EditImageModal(discord.ui.Modal, title="🖼️ Change Character Image"):
    image_url = discord.ui.TextInput(
        label="Image URL", placeholder="https://...", max_length=500, required=True
    )

    def __init__(self, mee_data: dict):
        super().__init__()
        self.mee_data = mee_data
        self.image_url.default = mee_data.get("image_url", DEFAULT_IMAGE)

    async def on_submit(self, interaction: discord.Interaction):
        await db.update_mee(self.mee_data["id"], image_url=self.image_url.value.strip())
        await interaction.response.send_message(
            embed=embeds.success_embed(
                f"**{self.mee_data['name']}**'s image updated!\n"
                "⚠️ Re-run `/mee webhook` to sync the webhook avatar."
            ),
            ephemeral=True,
        )


class EditNameModal(discord.ui.Modal, title="✏️ Rename Mee"):
    new_name = discord.ui.TextInput(label="New Name", max_length=32, required=True)

    def __init__(self, mee_data: dict):
        super().__init__()
        self.mee_data = mee_data
        self.new_name.default = mee_data["name"]

    async def on_submit(self, interaction: discord.Interaction):
        clash = await db.get_mee(self.new_name.value.strip())
        if clash and clash["id"] != self.mee_data["id"]:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"A Mee named **{self.new_name.value}** already exists."),
                ephemeral=True,
            )
            return
        await db.update_mee(self.mee_data["id"], name=self.new_name.value.strip())
        await interaction.response.send_message(
            embed=embeds.success_embed(f"Renamed to **{self.new_name.value}**!"),
            ephemeral=True,
        )


class EditTraitsModal(discord.ui.Modal, title="✨ Edit Traits & Goals"):
    traits = discord.ui.TextInput(
        label="Traits (comma-separated)", max_length=200, required=False
    )
    goals = discord.ui.TextInput(
        label="Goals (one per line)", style=discord.TextStyle.paragraph,
        max_length=400, required=False,
    )

    def __init__(self, mee_data: dict):
        super().__init__()
        self.mee_data = mee_data
        self.traits.default = ", ".join(mee_data.get("traits", []))
        self.goals.default  = "\n".join(mee_data.get("goals", []))

    async def on_submit(self, interaction: discord.Interaction):
        traits = [t.strip() for t in self.traits.value.split(",") if t.strip()]
        goals  = [g.strip() for g in self.goals.value.split("\n") if g.strip()]
        await db.update_mee(self.mee_data["id"], traits=traits, goals=goals)
        await interaction.response.send_message(
            embed=embeds.success_embed(f"**{self.mee_data['name']}**'s traits & goals updated!"),
            ephemeral=True,
        )


class WorldPostModal(discord.ui.Modal, title="🌍 Post World Update"):
    content = discord.ui.TextInput(
        label="World event description", style=discord.TextStyle.paragraph,
        placeholder="A thunderstorm rolls in. The power flickers...",
        max_length=500, required=True,
    )
    event_type = discord.ui.TextInput(
        label="Event type (update/event/movement/arrival)",
        placeholder="update", max_length=20, required=False,
    )

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        etype = self.event_type.value.strip() or "update"
        await db.add_world_event(etype, self.content.value.strip())
        embed = embeds.world_event_embed(self.content.value.strip(), etype)
        await interaction.channel.send(embed=embed)
        await interaction.followup.send(
            embed=embeds.success_embed("World update posted!"), ephemeral=True
        )


# ─── User /mymee create modal ─────────────────────────────────────────────────

class CreateMyMeeModal(discord.ui.Modal, title="🌸 Create Your Mee"):
    name = discord.ui.TextInput(
        label="Name", placeholder="e.g. Luna", max_length=32, required=True
    )
    identity = discord.ui.TextInput(
        label="Who are they? (personality, backstory)", style=discord.TextStyle.paragraph,
        placeholder="A dreamy poet who loves rainy days and strong tea...",
        max_length=1500, required=True,
    )
    traits = discord.ui.TextInput(
        label="Traits (comma-separated, optional)",
        placeholder="e.g. curious, sarcastic, warm-hearted",
        max_length=200, required=False,
    )
    goals = discord.ui.TextInput(
        label="Goals (one per line, optional)",
        style=discord.TextStyle.paragraph,
        placeholder="Find true connection\nLearn something new every day",
        max_length=400, required=False,
    )
    api_key_field = discord.ui.TextInput(
        label="Groq API Key (or API Key | Model | Base URL)",
        placeholder="gsk_xxxx... (free: console.groq.com) — leave blank if server provides one",
        max_length=400, required=False,
    )

    def __init__(self, channel_id: str):
        super().__init__()
        self.channel_id = channel_id

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        # Check permissions and quota
        if not await _check_create_permission(interaction):
            return

        # Parse API config
        raw = self.api_key_field.value.strip()
        if raw:
            parts    = [p.strip() for p in raw.split("|")]
            api_key  = parts[0]
            model    = parts[1] if len(parts) > 1 else DEFAULT_GROQ_MODEL
            api_base = parts[2] if len(parts) > 2 else DEFAULT_GROQ_BASE
        elif OWNER_API_KEY:
            api_key  = OWNER_API_KEY
            model    = OWNER_API_MODEL
            api_base = OWNER_API_BASE
        else:
            await interaction.followup.send(
                embed=embeds.error_embed(
                    "No API key provided and this server has no default key configured.\n\n"
                    "Get a **free** Groq key at [console.groq.com](https://console.groq.com) "
                    "and paste it in the API Key field."
                ),
                ephemeral=True,
            )
            return

        # Validate the API key
        await interaction.followup.send(
            embed=discord.Embed(description="🔍 Validating your API key...", color=0x888888),
            ephemeral=True,
        )
        valid = await validate_api_key(api_key, api_base, model)
        if not valid:
            await interaction.followup.send(
                embed=embeds.error_embed(
                    "❌ Could not validate this API key.\n\n"
                    "• Make sure the key is copied correctly\n"
                    "• For Groq: key starts with `gsk_`\n"
                    "• Base URL should be `https://api.groq.com/openai/v1`\n\n"
                    "Run `/mymee guide` for detailed instructions."
                ),
                ephemeral=True,
            )
            return

        name = self.name.value.strip()
        if await db.get_mee(name):
            await interaction.followup.send(
                embed=embeds.error_embed(f"A Mee named **{name}** already exists."),
                ephemeral=True,
            )
            return

        traits = [t.strip() for t in self.traits.value.split(",") if t.strip()] if self.traits.value else []
        goals  = [g.strip() for g in self.goals.value.split("\n") if g.strip()] if self.goals.value else []

        channel     = interaction.guild.get_channel(int(self.channel_id)) if interaction.guild else None
        webhook_url = await get_or_create_webhook(channel, interaction.client.user) if channel else None

        mee_id = await db.create_mee(
            name=name, identity=self.identity.value.strip(),
            traits=traits, goals=goals, model=model, api_key=api_key, api_base=api_base,
            image_url=DEFAULT_IMAGE, channel_id=self.channel_id, webhook_url=webhook_url,
            owner_discord_id=str(interaction.user.id),
            gemini_api_key=OWNER_GEMINI_API_KEY or None,
            gemini_model=OWNER_GEMINI_MODEL if OWNER_GEMINI_API_KEY else None,
        )

        mee   = await db.get_mee_by_id(mee_id)
        embed = embeds.mee_profile_embed(mee, memory_count=0)
        embed.title       = f"🌸 {name} is born! Welcome to the server."
        embed.description = (embed.description or "") + (
            "\n\n✅ Webhook configured — your Mee will chat with their own avatar!" if webhook_url
            else "\n\n⚠️ No webhook — ask a server admin to run `/mee webhook` for your Mee."
        )
        await interaction.followup.send(embed=embed, ephemeral=True)
        interaction.client.dispatch("mee_created", mee_id)


# ─── Management views ──────────────────────────────────────────────────────────

class ManageMeeView(discord.ui.View):
    """Full admin management panel."""
    def __init__(self, mee_data: dict):
        super().__init__(timeout=300)
        self.mee = mee_data

    @discord.ui.button(label="✏️ Identity", style=discord.ButtonStyle.primary, row=0)
    async def edit_identity(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditIdentityModal(self.mee))

    @discord.ui.button(label="✨ Traits & Goals", style=discord.ButtonStyle.primary, row=0)
    async def edit_traits(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditTraitsModal(self.mee))

    @discord.ui.button(label="🤖 LLM Settings", style=discord.ButtonStyle.secondary, row=1)
    async def edit_model(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditModelModal(self.mee))

    @discord.ui.button(label="🖼️ Image", style=discord.ButtonStyle.secondary, row=1)
    async def edit_image(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditImageModal(self.mee))

    @discord.ui.button(label="📝 Rename", style=discord.ButtonStyle.secondary, row=1)
    async def rename(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditNameModal(self.mee))

    @discord.ui.button(label="🌟 Foreground Key", style=discord.ButtonStyle.primary, row=1)
    async def edit_fg_key(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditForegroundKeyModal(self.mee))

    @discord.ui.button(label="🧠 Memories", style=discord.ButtonStyle.grey, row=2)
    async def view_memories(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        memories = await db.get_memories(self.mee["id"], limit=10)
        if not memories:
            await interaction.followup.send(
                embed=embeds.error_embed(f"{self.mee['name']} has no memories yet."), ephemeral=True)
            return
        embed = discord.Embed(
            title=f"🧠 {self.mee['name']}'s Recent Memories",
            color=embeds.mee_colour(self.mee["name"]),
        )
        for m in memories[:8]:
            icon = {"observation": "👁️", "reflection": "💭", "plan": "📋",
                    "conversation": "💬", "morning_recap": "🌅"}.get(m["memory_type"], "📝")
            val  = m["content"][:200] + ("..." if len(m["content"]) > 200 else "")
            embed.add_field(
                name=f"{icon} `{m['memory_type']}` · ⭐{m['importance']:.1f}",
                value=val, inline=False,
            )
        await interaction.followup.send(embed=embed, ephemeral=True)

    @discord.ui.button(label="💞 Relationships", style=discord.ButtonStyle.grey, row=2)
    async def view_relationships(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        rels  = await db.get_relationships(self.mee["id"])
        embed = discord.Embed(
            title=f"💞 {self.mee['name']}'s Relationships",
            color=embeds.mee_colour(self.mee["name"]),
        )
        if not rels:
            embed.description = "No relationships formed yet."
        for r in rels[:10]:
            sent_bar = "❤️" if r["sentiment"] > 0.5 else ("💔" if r["sentiment"] < -0.5 else "🤍")
            tier_str = f" [{r.get('tier','?')}]" if r.get("tier") else ""
            est_str  = " ❄️" if r.get("is_estranged") else ""
            embed.add_field(
                name=f"{sent_bar} {r['other_name']}{est_str}",
                value=f"*{r['relationship']}*{tier_str} · {r['sentiment']:+.2f}",
                inline=True,
            )
        await interaction.followup.send(embed=embed, ephemeral=True)

    @discord.ui.button(label="🧹 Clear All Memories", style=discord.ButtonStyle.danger, row=3)
    async def clear_all_memories(self, interaction: discord.Interaction, button: discord.ui.Button):
        count = await db.count_memories(self.mee["id"])
        view  = ConfirmMemoryClearView(self.mee, "all")
        await interaction.response.send_message(
            embed=discord.Embed(
                description=(
                    f"⚠️ Clear **ALL {count} memories** for **{self.mee['name']}**?\n"
                    "This will reset their entire memory and ChromaDB vectors. Cannot be undone."
                ),
                color=0xFF4444,
            ),
            view=view, ephemeral=True,
        )

    @discord.ui.button(label="📅 Clear Today", style=discord.ButtonStyle.secondary, row=3)
    async def clear_today_memories(self, interaction: discord.Interaction, button: discord.ui.Button):
        view = ConfirmMemoryClearView(self.mee, "today")
        await interaction.response.send_message(
            embed=discord.Embed(
                description=(
                    f"Clear **today's memories** for **{self.mee['name']}**?\n"
                    "Only memories from today will be deleted."
                ),
                color=0xFF8800,
            ),
            view=view, ephemeral=True,
        )

    @discord.ui.button(label="👤 Clear Person", style=discord.ButtonStyle.secondary, row=3)
    async def clear_person_memories(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(ClearPersonModal(self.mee))

    @discord.ui.button(label="�️ Remove", style=discord.ButtonStyle.danger, row=4)
    async def remove(self, interaction: discord.Interaction, button: discord.ui.Button):
        view = ConfirmDeleteView(self.mee)
        await interaction.response.send_message(
            embed=discord.Embed(
                description=f"⚠️ Remove **{self.mee['name']}**? This cannot be undone.",
                color=0xFF4444,
            ),
            view=view, ephemeral=True,
        )


class UserManageMeeView(discord.ui.View):
    """Simplified management view for non-admin Mee owners."""
    def __init__(self, mee_data: dict):
        super().__init__(timeout=300)
        self.mee = mee_data

    @discord.ui.button(label="✏️ Identity", style=discord.ButtonStyle.primary, row=0)
    async def edit_identity(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditIdentityModal(self.mee))

    @discord.ui.button(label="✨ Traits & Goals", style=discord.ButtonStyle.primary, row=0)
    async def edit_traits(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditTraitsModal(self.mee))

    @discord.ui.button(label="🤖 API Key / Model", style=discord.ButtonStyle.secondary, row=1)
    async def edit_model(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditModelModal(self.mee))

    @discord.ui.button(label="🖼️ Image URL", style=discord.ButtonStyle.secondary, row=1)
    async def edit_image(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditImageModal(self.mee))

    @discord.ui.button(label="🌟 Foreground Key", style=discord.ButtonStyle.primary, row=1)
    async def edit_fg_key(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EditForegroundKeyModal(self.mee))

    @discord.ui.button(label="🧹 Reset Memories", style=discord.ButtonStyle.danger, row=2)
    async def clear_user_memories(self, interaction: discord.Interaction, button: discord.ui.Button):
        count = await db.count_memories(self.mee["id"])
        view  = ConfirmMemoryClearView(self.mee, "all")
        await interaction.response.send_message(
            embed=discord.Embed(
                description=(
                    f"Reset **all {count} memories** for **{self.mee['name']}**?\n"
                    "Your character will start fresh. This cannot be undone."
                ),
                color=0xFF4444,
            ),
            view=view, ephemeral=True,
        )

    @discord.ui.button(label="🗑️ Delete", style=discord.ButtonStyle.danger, row=3)
    async def delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        view = ConfirmDeleteView(self.mee)
        await interaction.response.send_message(
            embed=discord.Embed(
                description=f"⚠️ Delete **{self.mee['name']}**? This cannot be undone.",
                color=0xFF4444,
            ),
            view=view, ephemeral=True,
        )


class ClearPersonModal(discord.ui.Modal, title="👤 Clear memories about a person"):
    person_name = discord.ui.TextInput(
        label="Person's name",
        placeholder="Name of the person to forget (exact or partial match)",
        max_length=64,
        required=True,
    )

    def __init__(self, mee_data: dict):
        super().__init__()
        self.mee = mee_data

    async def on_submit(self, interaction: discord.Interaction):
        name   = self.person_name.value.strip()
        ids    = await db.delete_memories_about_person(self.mee["id"], name)
        count  = len(ids)
        if ids:
            delete_memories_from_chroma(self.mee["id"], self.mee["name"], ids)
        await interaction.response.send_message(
            embed=discord.Embed(
                description=(
                    f"🧹 Cleared **{count}** memory{'s' if count != 1 else ''} mentioning "
                    f"**{name}** from **{self.mee['name']}**."
                    if count else
                    f"No memories mentioning **{name}** were found."
                ),
                color=0x00CC88 if count else 0x888888,
            ),
            ephemeral=True,
        )


class ConfirmMemoryClearView(discord.ui.View):
    """Confirmation dialog for memory clearing operations."""
    def __init__(self, mee_data: dict, clear_type: str):
        super().__init__(timeout=30)
        self.mee        = mee_data
        self.clear_type = clear_type  # "all" | "today"

    @discord.ui.button(label="Yes, clear", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        name = self.mee["name"]
        mee_id = self.mee["id"]

        if self.clear_type == "all":
            count = await db.delete_all_memories(mee_id)
            delete_collection(mee_id, name)
            desc = f"🧹 Cleared **all {count} memories** for **{name}**. They start fresh."
        else:  # today
            count = await db.delete_today_memories(mee_id)
            # For "today" wipe, re-query ids that were deleted — just purge & re-sync
            # ChromaDB will naturally lose them on next sync; no IDs to pass
            try:
                delete_collection(mee_id, name)
            except Exception:
                pass
            desc = f"📅 Cleared **{count}** of today's memories for **{name}**."

        # Add a soft reset observation so the Mee knows something changed
        await db.add_memory(
            mee_id,
            f"I feel like I've forgotten something... my memories feel hazy.",
            "observation",
            importance=4.0,
        )
        await interaction.response.edit_message(
            embed=discord.Embed(description=desc, color=0x00CC88), view=None
        )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(
            embed=embeds.success_embed("Cancelled."), view=None
        )


class ConfirmDeleteView(discord.ui.View):
    def __init__(self, mee_data: dict):
        super().__init__(timeout=30)
        self.mee = mee_data

    @discord.ui.button(label="Yes, remove", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await db.delete_mee(self.mee["name"])
        interaction.client.dispatch("mee_removed", self.mee["id"])
        await interaction.response.edit_message(
            embed=embeds.success_embed(f"**{self.mee['name']}** has been removed."), view=None)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(
            embed=embeds.success_embed("Cancelled."), view=None)


# ─── Cog ───────────────────────────────────────────────────────────────────────

class ManageCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # ═══════════════════════════════════════════════════════════════════════════
    # /mee group — OWNER ONLY admin commands
    # ═══════════════════════════════════════════════════════════════════════════

    mee_group = app_commands.Group(name="mee", description="Admin: manage all Mees")

    @mee_group.command(name="list", description="List all Mees")
    @is_owner()
    async def mee_list(self, interaction: discord.Interaction):
        mees  = await db.list_mees()
        embed = embeds.list_mees_embed(mees)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @mee_group.command(name="add", description="Create a new Mee (admin — full config)")
    @is_owner()
    async def mee_add(self, interaction: discord.Interaction):
        await interaction.response.send_modal(AddMeeModal(str(interaction.channel_id)))

    @mee_group.command(name="remove", description="Remove a Mee")
    @app_commands.describe(name="Name of the Mee to remove")
    @is_owner()
    async def mee_remove(self, interaction: discord.Interaction, name: str):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        view = ConfirmDeleteView(mee)
        await interaction.response.send_message(
            embed=discord.Embed(description=f"⚠️ Remove **{mee['name']}**?", color=0xFF4444),
            view=view, ephemeral=True,
        )

    @mee_group.command(name="manage", description="Open full management panel for a Mee")
    @app_commands.describe(name="Name of the Mee")
    @is_owner()
    async def mee_manage(self, interaction: discord.Interaction, name: str):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        memories = await db.get_memories(mee["id"], limit=200)
        embed    = embeds.mee_profile_embed(mee, memory_count=len(memories))
        view     = ManageMeeView(mee)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    # ── Status and mood defined BEFORE autocomplete registration ──────────────

    @mee_group.command(name="status", description="Live status card: location, mood, excitement")
    @app_commands.describe(name="Name of the Mee")
    @is_owner()
    async def mee_status(self, interaction: discord.Interaction, name: str):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        channel_id = mee.get("channel_id", "")
        excitement = getattr(interaction.client, "excitement", {}).get(channel_id, 0.0)
        embed      = embeds.mee_status_embed(mee, excitement)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @mee_group.command(name="mood", description="View or override a Mee's current mood")
    @app_commands.describe(name="Name of the Mee", mood="New mood (leave blank to view)")
    @is_owner()
    async def mee_mood(self, interaction: discord.Interaction, name: str, mood: str = ""):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        if not mood:
            current = mee.get("mood", "neutral")
            embed   = discord.Embed(
                description=f"**{name}** is currently feeling: *{current}*",
                color=embeds.mee_colour(name),
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            mood = mood.strip()[:60]
            await db.update_mee(mee["id"], mood=mood)
            agent = interaction.client.agents.get(mee["id"])
            if agent:
                agent.mood = mood
            await interaction.response.send_message(
                embed=embeds.success_embed(f"**{name}**'s mood set to: *{mood}*"), ephemeral=True
            )

    # ── Autocomplete registration (all commands defined above this line) ───────

    @mee_manage.autocomplete("name")
    @mee_remove.autocomplete("name")
    @mee_status.autocomplete("name")
    @mee_mood.autocomplete("name")
    async def mee_name_autocomplete(self, interaction: discord.Interaction,
                                     current: str) -> list[app_commands.Choice]:
        mees = await db.list_mees()
        return [
            app_commands.Choice(name=m["name"], value=m["name"])
            for m in mees if current.lower() in m["name"].lower()
        ][:25]

    # ── Remaining admin commands ───────────────────────────────────────────────

    @mee_group.command(name="summon", description="Force a Mee to speak right now")
    @app_commands.describe(name="Name of the Mee")
    @is_owner()
    async def mee_summon(self, interaction: discord.Interaction, name: str):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        await interaction.response.send_message(embed=embeds.thinking_embed(name), ephemeral=True)
        interaction.client.dispatch("mee_force_speak", mee["id"], interaction.channel_id)

    @mee_group.command(name="channel", description="Change which channel a Mee lives in")
    @app_commands.describe(name="Name of the Mee", channel="The channel")
    @is_owner()
    async def mee_channel(self, interaction: discord.Interaction, name: str,
                           channel: discord.TextChannel):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        webhook_url = await get_or_create_webhook(channel, interaction.client.user)
        await db.update_mee(mee["id"], channel_id=str(channel.id), webhook_url=webhook_url)
        msg = f"**{name}** will now live in {channel.mention}!"
        if webhook_url:
            msg += " Webhook auto-configured ✅"
        await interaction.response.send_message(embed=embeds.success_embed(msg), ephemeral=True)

    @mee_group.command(name="webhook", description="Set up or refresh webhook for a Mee")
    @app_commands.describe(name="Name of the Mee")
    @is_owner()
    async def mee_webhook(self, interaction: discord.Interaction, name: str):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        channel = interaction.guild.get_channel(int(mee["channel_id"])) if mee.get("channel_id") else None
        if not channel:
            await interaction.followup.send(
                embed=embeds.error_embed("Mee has no channel set. Use `/mee channel` first."),
                ephemeral=True)
            return
        webhook_url = await get_or_create_webhook(channel, interaction.client.user)
        if webhook_url:
            await db.update_mee(mee["id"], webhook_url=webhook_url)
            await interaction.followup.send(
                embed=embeds.success_embed(f"Webhook configured for **{name}** in {channel.mention}!"),
                ephemeral=True)
        else:
            await interaction.followup.send(
                embed=embeds.error_embed("Failed to create webhook. Check Manage Webhooks permission."),
                ephemeral=True)

    @mee_group.command(name="move", description="Move a Mee to a new location (posts world update)")
    @app_commands.describe(name="Name of the Mee", location="New location")
    @is_owner()
    async def mee_move(self, interaction: discord.Interaction, name: str, location: str):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        agent = interaction.client.agents.get(mee["id"])
        if agent:
            event = await agent.move_to(location)
        else:
            await db.update_mee(mee["id"], location=location)
            event = f"{name} moved to {location}."
            await db.add_world_event("movement", event)
        embed = embeds.world_event_embed(event, "movement")
        await interaction.channel.send(embed=embed)
        await interaction.followup.send(
            embed=embeds.success_embed(f"**{name}** moved to *{location}*."), ephemeral=True)

    @mee_group.command(name="memory", description="View a Mee's recent memories")
    @app_commands.describe(name="Name of the Mee")
    @is_owner()
    async def mee_memory(self, interaction: discord.Interaction, name: str):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        memories = await db.get_memories(mee["id"], limit=12)
        if not memories:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"**{name}** has no memories yet."), ephemeral=True)
            return
        embed = discord.Embed(
            title=f"🧠 {mee['name']}'s Memory Stream",
            color=embeds.mee_colour(mee["name"]),
            description=f"Showing {len(memories)} most recent",
        )
        embed.set_thumbnail(url=mee.get("image_url") or DEFAULT_IMAGE)
        for m in memories[:10]:
            icon = {"observation": "👁️", "reflection": "💭", "plan": "📋",
                    "conversation": "💬", "morning_recap": "🌅"}.get(m["memory_type"], "📝")
            val  = m["content"][:180] + ("..." if len(m["content"]) > 180 else "")
            embed.add_field(
                name=f"{icon} {m['memory_type']} · ⭐{m['importance']:.1f} · {m['created_at'][:16]}",
                value=val, inline=False,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @mee_group.command(name="relationships", description="View a Mee's relationships")
    @app_commands.describe(name="Name of the Mee")
    @is_owner()
    async def mee_relationships(self, interaction: discord.Interaction, name: str):
        mee = await db.get_mee(name)
        if not mee:
            await interaction.response.send_message(
                embed=embeds.error_embed(f"No Mee named **{name}** found."), ephemeral=True)
            return
        rels  = await db.get_relationships(mee["id"])
        embed = discord.Embed(title=f"💞 {mee['name']}'s Relationships", color=embeds.mee_colour(mee["name"]))
        embed.set_thumbnail(url=mee.get("image_url") or DEFAULT_IMAGE)
        if not rels:
            embed.description = "No relationships formed yet."
        for r in rels[:12]:
            sent_bar = "❤️" if r["sentiment"] > 0.5 else ("💔" if r["sentiment"] < -0.5 else "🤍")
            tier_str = f" [{r.get('tier','?')}]"
            est_str  = " ❄️" if r.get("is_estranged") else ""
            embed.add_field(
                name=f"{sent_bar} {r['other_name']}{est_str}",
                value=f"*{r['relationship']}*{tier_str} · {r['sentiment']:+.2f}",
                inline=True,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @mee_group.command(name="world", description="View recent world events")
    @is_owner()
    async def mee_world(self, interaction: discord.Interaction):
        events = await db.get_recent_world_events(limit=15)
        embed  = discord.Embed(title="🌍 World State", color=0x888888)
        if not events:
            embed.description = "Nothing has happened yet."
        for e in events[-10:]:
            embed.add_field(
                name=f"{e['event_type']} · {e['created_at'][:16]}",
                value=e["content"][:200], inline=False,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @mee_group.command(name="world-post", description="Post a world update to the channel")
    @is_owner()
    async def mee_world_post(self, interaction: discord.Interaction):
        await interaction.response.send_modal(WorldPostModal())

    @mee_group.command(name="locations", description="View or set the wander location list")
    @app_commands.describe(locations="Comma-separated list (leave blank to view)")
    @is_owner()
    async def mee_locations(self, interaction: discord.Interaction, locations: str = ""):
        guild_id = str(interaction.guild_id)
        if not locations:
            from src.agents.agent import DEFAULT_LOCATIONS
            current = await db.get_server_locations(guild_id) or DEFAULT_LOCATIONS
            label   = "*(custom)*" if await db.get_server_locations(guild_id) else "*(defaults)*"
            embed   = discord.Embed(
                title="📍 Wander Locations", color=0x88BBFF,
                description=f"{label}\n\n" + "\n".join(f"• {loc}" for loc in current),
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            locs = [l.strip() for l in locations.split(",") if l.strip()]
            if len(locs) < 2:
                await interaction.response.send_message(
                    embed=embeds.error_embed("Provide at least 2 locations."), ephemeral=True)
                return
            await db.set_server_locations(guild_id, locs)
            await interaction.response.send_message(
                embed=embeds.success_embed(
                    "Updated! Mees will wander between:\n" + "\n".join(f"• {l}" for l in locs)
                ),
                ephemeral=True,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # /mymee group — ANY USER commands (scoped to their own Mee)
    # ═══════════════════════════════════════════════════════════════════════════

    mymee_group = app_commands.Group(
        name="mymee",
        description="Create and manage your own Mee character",
    )

    @mymee_group.command(name="create", description="Create your own Mee character")
    async def mymee_create(self, interaction: discord.Interaction):
        await interaction.response.send_modal(CreateMyMeeModal(str(interaction.channel_id)))

    @mymee_group.command(name="guide", description="How to get a free Groq API key for your Mee")
    async def mymee_guide(self, interaction: discord.Interaction):
        await interaction.response.send_message(embed=groq_onboarding_embed(), ephemeral=True)

    @mymee_group.command(name="manage", description="Edit your Mee character")
    async def mymee_manage(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        my_mees = await db.list_mees_by_owner(str(interaction.user.id))
        if not my_mees:
            await interaction.followup.send(
                embed=embeds.error_embed(
                    "You don't have a Mee yet. Use `/mymee create` to make one!"
                ),
                ephemeral=True,
            )
            return
        mee      = my_mees[0]
        memories = await db.get_memories(mee["id"], limit=200)
        embed    = embeds.mee_profile_embed(mee, memory_count=len(memories))
        view     = UserManageMeeView(mee)
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    @mymee_group.command(name="status", description="View your Mee's live status")
    async def mymee_status(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        my_mees = await db.list_mees_by_owner(str(interaction.user.id))
        if not my_mees:
            await interaction.followup.send(
                embed=embeds.error_embed("You don't have a Mee yet."), ephemeral=True)
            return
        mee        = my_mees[0]
        channel_id = mee.get("channel_id", "")
        excitement = getattr(interaction.client, "excitement", {}).get(channel_id, 0.0)
        embed      = embeds.mee_status_embed(mee, excitement)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @mymee_group.command(name="summon", description="Make your Mee speak right now")
    async def mymee_summon(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        my_mees = await db.list_mees_by_owner(str(interaction.user.id))
        if not my_mees:
            await interaction.followup.send(
                embed=embeds.error_embed("You don't have a Mee yet."), ephemeral=True)
            return
        mee = my_mees[0]
        await interaction.followup.send(embed=embeds.thinking_embed(mee["name"]), ephemeral=True)
        interaction.client.dispatch("mee_force_speak", mee["id"], interaction.channel_id)

    @mymee_group.command(name="memory", description="View your Mee's recent memories")
    async def mymee_memory(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        my_mees = await db.list_mees_by_owner(str(interaction.user.id))
        if not my_mees:
            await interaction.followup.send(
                embed=embeds.error_embed("You don't have a Mee yet."), ephemeral=True)
            return
        mee      = my_mees[0]
        memories = await db.get_memories(mee["id"], limit=10)
        if not memories:
            await interaction.followup.send(
                embed=embeds.error_embed(f"**{mee['name']}** has no memories yet."), ephemeral=True)
            return
        embed = discord.Embed(
            title=f"🧠 {mee['name']}'s Memory Stream",
            color=embeds.mee_colour(mee["name"]),
        )
        for m in memories[:8]:
            icon = {"observation": "👁️", "reflection": "💭", "plan": "📋",
                    "conversation": "💬", "morning_recap": "🌅"}.get(m["memory_type"], "📝")
            val  = m["content"][:180] + ("..." if len(m["content"]) > 180 else "")
            embed.add_field(
                name=f"{icon} {m['memory_type']} · ⭐{m['importance']:.1f}",
                value=val, inline=False,
            )
        await interaction.followup.send(embed=embed, ephemeral=True)

    @mymee_group.command(name="delete", description="Delete your Mee character")
    async def mymee_delete(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        my_mees = await db.list_mees_by_owner(str(interaction.user.id))
        if not my_mees:
            await interaction.followup.send(
                embed=embeds.error_embed("You don't have a Mee yet."), ephemeral=True)
            return
        mee  = my_mees[0]
        view = ConfirmDeleteView(mee)
        await interaction.followup.send(
            embed=discord.Embed(
                description=f"⚠️ Delete **{mee['name']}**? This cannot be undone.",
                color=0xFF4444,
            ),
            view=view, ephemeral=True,
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(ManageCog(bot))

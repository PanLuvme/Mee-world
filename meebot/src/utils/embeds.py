"""
Discord embed helpers — management panels and world events.
Mee messages are posted via webhook (plain text with their PFP).

v5 changes:
- Updated DEFAULT_IMAGE to new default Mee avatar
- Full v4 EVENT_ICONS map in world_event_embed()
- groq_onboarding_embed() for new user guidance
- mee_status_embed shows relationship tier counts
"""
import discord
from datetime import datetime

DEFAULT_IMAGE = "https://i.postimg.cc/cJnwz3qf/Default-Mee.png"

# Full event type → emoji map (covers v2 through v4 event types)
EVENT_ICONS = {
    "movement":       "🗺️",
    "arrival":        "✨",
    "departure":      "👋",
    "update":         "🌍",
    "event":          "⚡",
    "relationship":   "💞",
    "fight":          "⚡",
    "reconciliation": "💚",
    "crush":          "💓",
    "confession":     "💌",
    "introduction":   "🤝",
    "need":           "🫀",
    "morning_recap":  "🌅",
}


def mee_colour(name: str) -> int:
    colours = [
        0xE91E8C, 0x00B0FF, 0xFF6D00, 0x69F0AE, 0xEA80FC,
        0xFFD740, 0x40C4FF, 0xFF6E40, 0xB2FF59, 0xFF4081,
    ]
    return colours[hash(name) % len(colours)]


def list_mees_embed(mees: list[dict]) -> discord.Embed:
    embed = discord.Embed(
        title="🌸 Active Mees",
        description=f"**{len(mees)}** Mee(s) living in this server.",
        color=0xE91E8C,
        timestamp=datetime.utcnow(),
    )
    for mee in mees:
        traits       = ", ".join(mee["traits"][:3]) if mee["traits"] else "mysterious"
        wh_status    = "✅" if mee.get("webhook_url") else "⚠️"
        channel      = f"<#{mee['channel_id']}>" if mee.get("channel_id") else "none"
        owner_tag    = f"<@{mee['owner_discord_id']}>" if mee.get("owner_discord_id") and mee["owner_discord_id"] != "0" else "server"
        val = (
            f"🤖 `{mee['model']}`\n"
            f"✨ *{traits}*\n"
            f"📍 {channel}\n"
            f"🗺️ *{mee.get('location', '?')}*\n"
            f"😶 *{mee.get('mood', 'neutral')}*\n"
            f"🔗 webhook {wh_status}\n"
            f"👤 {owner_tag}"
        )
        embed.add_field(name=f"**{mee['name']}**", value=val, inline=True)
    if not mees:
        embed.description = "No Mees yet! Use `/mee add` (admin) or `/mymee create` (any user) to create the first one."
    embed.set_footer(text="/mee manage <name> to edit • /mymee create to make your own")
    return embed


def mee_profile_embed(mee: dict, memory_count: int = 0) -> discord.Embed:
    embed = discord.Embed(
        title=f"🌸 {mee['name']}",
        description=mee["identity"][:500] + ("..." if len(mee["identity"]) > 500 else ""),
        color=mee_colour(mee["name"]),
        timestamp=datetime.utcnow(),
    )
    embed.set_thumbnail(url=mee.get("image_url") or DEFAULT_IMAGE)

    traits      = ", ".join(mee["traits"]) if mee["traits"] else "none set"
    goals       = "\n".join(f"• {g}" for g in mee["goals"][:3]) if mee["goals"] else "• none set"
    channel     = f"<#{mee['channel_id']}>" if mee.get("channel_id") else "none"
    webhook     = "✅ configured" if mee.get("webhook_url") else "⚠️ not set (use /mee webhook)"
    owner_tag   = f"<@{mee['owner_discord_id']}>" if mee.get("owner_discord_id") and mee["owner_discord_id"] != "0" else "server admin"

    embed.add_field(name="✨ Traits",   value=traits,    inline=True)
    embed.add_field(name="🎯 Goals",   value=goals,     inline=True)
    embed.add_field(name="🤖 Model",   value=f"`{mee['model']}`", inline=True)
    embed.add_field(name="📍 Channel", value=channel,   inline=True)
    embed.add_field(name="🗺️ Location",value=mee.get("location", "?"), inline=True)
    embed.add_field(name="😶 Mood",    value=mee.get("mood", "neutral"), inline=True)
    embed.add_field(name="🔗 Webhook", value=webhook,   inline=True)
    embed.add_field(name="🧠 Memories",value=str(memory_count), inline=True)
    embed.add_field(name="👤 Owner",   value=owner_tag, inline=True)
    embed.set_footer(text=f"ID: {mee['id']} • Born: {mee.get('created_at','?')[:10]}")
    return embed


def world_event_embed(content: str, event_type: str = "update") -> discord.Embed:
    """Embed for world state announcements posted by the main bot."""
    icon = EVENT_ICONS.get(event_type, "🌍")
    embed = discord.Embed(
        description=f"{icon} *{content}*",
        color=0x888888,
        timestamp=datetime.utcnow(),
    )
    embed.set_footer(text="World update")
    return embed


def mee_status_embed(mee: dict, excitement: float = 0.0) -> discord.Embed:
    """Compact at-a-glance status card for a Mee."""
    mood      = mee.get("mood", "neutral")
    location  = mee.get("location", "somewhere")
    last_tick = mee.get("last_tick", "never")[:16] if mee.get("last_tick") else "never"
    excite_bar = "🟡" * int(excitement * 5) + "⬜" * (5 - int(excitement * 5))
    embed = discord.Embed(
        title=f"📡 {mee['name']} — live status",
        color=mee_colour(mee["name"]),
        timestamp=datetime.utcnow(),
    )
    embed.set_thumbnail(url=mee.get("image_url") or DEFAULT_IMAGE)
    embed.add_field(name="🗺️ Location",        value=location,    inline=True)
    embed.add_field(name="😶 Mood",            value=mood,        inline=True)
    embed.add_field(name="⚡ Excitement",      value=excite_bar,  inline=True)
    embed.add_field(name="🕐 Last spoke",      value=last_tick,   inline=True)
    embed.add_field(name="Status",             value="✅ active" if mee.get("active") else "💤 inactive", inline=True)
    return embed


def groq_onboarding_embed() -> discord.Embed:
    """Explain how to get a free Groq API key — shown when /mymee create is run."""
    embed = discord.Embed(
        title="🔑 How to get a free Groq API key",
        description=(
            "Groq provides **free API access** to fast LLMs — perfect for powering your Mee.\n\n"
            "**Steps:**\n"
            "1. Go to [console.groq.com](https://console.groq.com)\n"
            "2. Sign up / log in (free)\n"
            "3. Click **API Keys** → **Create API Key**\n"
            "4. Copy the key (starts with `gsk_...`)\n\n"
            "**Recommended model:** `llama-3.1-8b-instant`\n"
            "**Base URL:** `https://api.groq.com/openai/v1`\n\n"
            "**Free tier limits:**\n"
            "• 14,400 requests/day\n"
            "• 500 requests/minute\n"
            "• More than enough for one active Mee 🌸\n\n"
            "Then run `/mymee create` and paste your key in the **API Key** field."
        ),
        color=0xFF6B35,
    )
    embed.set_footer(text="OpenRouter, OpenAI, Mistral, Ollama also work — any OpenAI-compatible endpoint")
    return embed


def error_embed(message: str) -> discord.Embed:
    return discord.Embed(description=f"❌ {message}", color=0xFF4444)


def success_embed(message: str) -> discord.Embed:
    return discord.Embed(description=f"✅ {message}", color=0x44FF88)


def thinking_embed(name: str) -> discord.Embed:
    return discord.Embed(description=f"*{name} is thinking...*", color=0x888888)

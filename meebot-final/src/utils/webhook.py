"""
Webhook poster — Mees post as themselves via Discord webhooks.

v5 changes:
- Accept an optional shared aiohttp.ClientSession (bot-lifetime reuse).
  If none is given, a one-off session is used (backward compatible).
- Removed per-call session creation from the hot path.
"""
import aiohttp
import logging
from typing import Optional

import discord

logger = logging.getLogger(__name__)


async def post_as_mee(
    channel: discord.TextChannel,
    name: str,
    content: str,
    image_url: Optional[str],
    webhook_url: Optional[str],
    session: Optional[aiohttp.ClientSession] = None,
) -> bool:
    """
    Post a message as the Mee character.
    Returns True if webhook was used, False if fell back to channel.send.
    """
    if webhook_url:
        success = await _post_via_webhook(webhook_url, name, content, image_url, session)
        if success:
            return True
        logger.warning(f"[{name}] Webhook post failed, falling back to channel.send")

    try:
        await channel.send(f"**{name}**: {content}")
    except Exception as e:
        logger.error(f"[{name}] channel.send failed: {e}")
    return False


async def _post_via_webhook(
    webhook_url: str,
    username: str,
    content: str,
    avatar_url: Optional[str],
    session: Optional[aiohttp.ClientSession] = None,
) -> bool:
    """Send a message through a Discord webhook, reusing the bot's session if provided."""
    payload: dict = {"content": content, "username": username}
    if avatar_url:
        payload["avatar_url"] = avatar_url

    async def _do_post(s: aiohttp.ClientSession) -> bool:
        try:
            async with s.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status in (200, 204):
                    return True
                body = await resp.text()
                logger.warning(f"Webhook returned {resp.status}: {body[:200]}")
                return False
        except Exception as e:
            logger.warning(f"Webhook request failed: {e}")
            return False

    if session and not session.closed:
        return await _do_post(session)

    # Fallback: one-off session (used if no shared session provided)
    async with aiohttp.ClientSession() as one_off:
        return await _do_post(one_off)


async def get_or_create_webhook(
    channel: discord.TextChannel,
    bot_user: discord.ClientUser,
) -> Optional[str]:
    """
    Find an existing MeeBot webhook in the channel, or create one.
    Returns the webhook URL, or None on failure.
    """
    try:
        webhooks = await channel.webhooks()
        for wh in webhooks:
            if wh.name == "MeeBot" and wh.user == bot_user:
                return wh.url
        wh = await channel.create_webhook(name="MeeBot")
        logger.info(f"Created new webhook in #{channel.name}: {wh.id}")
        return wh.url
    except discord.Forbidden:
        logger.warning(f"No permission to manage webhooks in #{channel.name}")
        return None
    except Exception as e:
        logger.error(f"Error managing webhooks in #{channel.name}: {e}")
        return None

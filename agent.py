import logging

from dotenv import load_dotenv
from typing import Annotated
import aiohttp

from livekit.agents import llm
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero
from livekit.plugins import turn_detector
from tools import AssistantFnc

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant that can help with a variety of tasks. "
            "keep your responses short and to the point. "
        ),
    )
    # first define a class that inherits from llm.FunctionContext
   

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    # initialize the AssistantFnc class and pass it to the VoicePipelineAgent
    func_ctx = AssistantFnc()
    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # This project is configured to use Deepgram STT, OpenAI LLM and TTS plugins
    # Other great providers exist like Cartesia and ElevenLabs
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        # turn_detector=turn_detector.EOUModel(),
        stt=openai.STT.with_groq(model="whisper-large-v3-turbo"),
        llm=openai.LLM.with_groq(model="llama-3.3-70b-versatile"),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=func_ctx,
        max_nested_fnc_calls=5
    )

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )

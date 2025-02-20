import logging

from dotenv import load_dotenv
from typing import Annotated
from livekit.plugins import turn_detector
from livekit.plugins.elevenlabs import tts
import asyncio
from livekit import rtc
from livekit.agents import stt, transcription
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

async def _forward_transcription(
    stt_stream: stt.SpeechStream,
    stt_forwarder: transcription.STTSegmentsForwarder,
):
    """Forward the transcription and log the transcript in the console"""
    async for ev in stt_stream:
        stt_forwarder.update(ev)
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            print(ev.alternatives[0].text, end="")
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            print("\n")
            print(" -> ", ev.alternatives[0].text)


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            """you are a helpful assistant"""
        ),
    )
    
    stt = openai.STT()
    tasks = []

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stt_forwarder = transcription.STTSegmentsForwarder(
            room=ctx.room, participant=participant, track=track
        )
        stt_stream = stt.stream()
        stt_task = asyncio.create_task(
            _forward_transcription(stt_stream, stt_forwarder)
        )
        print(f"Transcription: {stt_task}")
        tasks.append(stt_task)

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            tasks.append(asyncio.create_task(transcribe_track(participant, track)))
        logger.error("subscribed to track")
    
    
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    # initialize the AssistantFnc class and pass it to the VoicePipelineAgent
    func_ctx = AssistantFnc()
    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")
    eleven_tts=tts.TTS(
        model="eleven_turbo_v2_5",
        api_key="sk_783fc29124493029c3553a30dfc3446998f91de38e814118",
        voice=tts.Voice(
            id="A9ATTqUUQ6GHu0coCz8t",
            name="Hamid",
            category="premade",
            settings=tts.VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            ),
        ),
        language="ar",
        streaming_latency=3,
        enable_ssml_parsing=False,
        chunk_length_schedule=[80, 120, 200, 260],
    )
    # This project is configured to use Deepgram STT, OpenAI LLM and TTS plugins
    # Other great providers exist like Cartesia and ElevenLabs
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=eleven_tts,
        chat_ctx=initial_ctx,
        fnc_ctx=func_ctx,
        max_nested_fnc_calls=5,
        turn_detector=turn_detector.EOUModel(),
    )

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hey, how can help you today", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )

import enum
from typing import Annotated
from livekit.agents import llm
import logging
import aiohttp
import json

logger = logging.getLogger("weather-agent")
logger.setLevel(logging.INFO)


class Zone(enum.Enum):
    LIVING_ROOM = "living_room"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    OFFICE = "office"

class AssistantFnc(llm.FunctionContext):
    # the __init__ function is called when the class is instantiated
    def __init__(self, ctx , participant) -> None:
        self.ctx = ctx
        self.participant = participant
        super().__init__()

        self.temperature = {
            Zone.LIVING_ROOM: 72,
            Zone.KITCHEN: 68,
            Zone.BEDROOM: 70,
            Zone.BATHROOM: 75,
            Zone.OFFICE: 73,
        }
    # the llm.ai_callable decorator marks this function as a tool available to the LLM
    # by default, it'll use the docstring as the function's description
    @llm.ai_callable()
    async def get_weather(
        self,
        # by using the Annotated type, arg description and type are available to the LLM
        location: Annotated[
            str, llm.TypeInfo(description="The location to get the weather for")
        ],
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        logger.info(f"getting weather for {location}")
        url = f"https://wttr.in/{location}?format=%C+%t"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    weather_data = await response.text()
                    # response from the function call is returned to the LLM
                    # as a tool response. The LLM's response will include this data
                    return f"The weather in {location} is {weather_data}."
                else:
                    raise f"Failed to get weather data, status code: {response.status}"
    @llm.ai_callable()
    async def send_email(
        self,
        email: Annotated[
            str, llm.TypeInfo(description="The email address to send the email to")
        ],
        message: Annotated[
            str, llm.TypeInfo(description="The message to include in the email")
        ],
    ):
        """Called when the user asks to send an email. This function will send an email to the given email address."""
        logger.info(f"sending email to {email}")
        # send email implementation would go here
        return f"Email sent to {email} with message: {message}"
    
    @llm.ai_callable()
    async def get_temperature(
        self,
        zone: Annotated[
            Zone, llm.TypeInfo(description="The zone to get the temperature for")
        ],
    ):
        """Called when the user asks about the temperature of the room. This function will return the temperature for the given location."""
        logger.info(f"getting temperature for {zone}")
        temp = self.temperature[Zone(zone)]
        return f"The temperature in the {zone} is {temp} degrees C."
    
    @llm.ai_callable()
    async def get_user_location(self,
        high_accuracy: Annotated[
            bool, llm.TypeInfo(description="Whether to use high accuracy mode, which is slower")
        ] = False
    ):
        """Retrieve the user's current geolocation as lat/lng."""
        try:
            return await self.ctx.room.local_participant.perform_rpc(
                destination_identity=self.participant.identity,
                method="getUserLocation",
                payload=json.dumps({
                    "highAccuracy": high_accuracy
                }),
                response_timeout=10.0 if high_accuracy else 5.0,
            )
        except Exception:
            return "Unable to retrieve user location"

    
    
    @llm.ai_callable()
    async def get_news(self):
        """Called when the user asks for the latest news. This function will return the latest news headlines."""
        logger.info("getting news")
        # get news implementation would go here
        news =  """Federal government workers have been left “shell-shocked” by the upheaval wreaked by Donald Trump’s return to the presidency amid signs that he is bent on exacting revenge on a bureaucracy he considers to be a “deep state” that previously thwarted and persecuted him.

Since being restored to the White House on 20 January, the president has gone on a revenge spree against high-profile figures who previously served him but earned his enmity by slighting or criticising him in public.

He has cancelled Secret Service protection for three senior national security officials in his first presidency – John Bolton, the former national security adviser; Mike Pompeo, who was CIA director and secretary of state; and Brian Hook, a former assistant secretary of state – even though all are assassination targets on an Iranian government hit list.

The same treatment has been meted out to Anthony Fauci, the infectious diseases expert who angered Trump after joining the White House taskforce tackling Covid-19 and who has also faced death threats.

Trump has also fired high-profile figures from government roles on his social media site and stripped 51 former intelligence officials of their security clearances for doubting reports about Hunter Biden’s laptop as possible Russian disinformation.

Yet whereas Trump’s better-known adversaries were possibly expecting a measure of payback – and in some cases, like Fauci’s, were pardoned by Joe Biden to shield them from prosecution – more intense vengeance may have been felt by anonymous civil servants who were less prepared.

Some senior officials saw the writing on the wall and resigned before his return, but others adopted a hope-for-the-best attitude – only to be shocked by what awaited them, according to insiders.

a silhouetted government building
Federal workers decry Trump attempt to force mass resignations as ‘cruel joke’
Read more"""

        return await self.ctx.room.local_participant.perform_rpc(
                destination_identity=self.participant.identity,
                method="get_news",
                payload=json.dumps({
                    "news": news
                }),
            )
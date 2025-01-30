import enum
from typing import Annotated
from livekit.agents import llm
import logging
import aiohttp

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
    def __init__(self) -> None:
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
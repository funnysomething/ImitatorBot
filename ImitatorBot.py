import discord
from discord.ext import commands
from mlBot.mlBot import generate_text
from mlBot.mlBot import load_model
from mlBot.mlBot import TextGenerationModel
from chatGPTBot.chatgptimitator import generate_response
import configparser

#Notifies discord API that bot wants messages and message content from server
intents = discord.Intents.default()
intents.message_content = True

#Creates bot with given command prefix
bot = commands.Bot(command_prefix='!', intents=intents)

#Gets bot_token from config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
bot_token = config['Credentials']['DISCORD_BOT_TOKEN']

model, word_to_index = load_model()

person = "Barack Obama"

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.event
async def on_message(message):
    print(f"{message.author.name}: {message.content}")
    if message.author == bot.user:
        return
    await bot.process_commands(message)

@bot.command()
async def generate(ctx, *, arg):
    print("Generating response")
    global model
    global word_to_index
    response = generate_text(model=model, word_to_index=word_to_index, seed_text=arg,max_length=14)
    await ctx.send(response)

@bot.command()
async def imitate(ctx, *, arg):
    global person
    person = arg
    response = await generate_response(person=person, message="hi")
    await ctx.send(response)

@bot.command()
async def test(ctx, *, arg):
    await ctx.send(arg)

bot.run(bot_token)
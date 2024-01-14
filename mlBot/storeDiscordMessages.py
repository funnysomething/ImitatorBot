import sqlite3
import discord
from discord.ext import commands

bot_token = 'MTE5MzcwNTczMTMyOTkwMDYwNA.GXhE0y._D7d_M_pXy08rQKBzZXrAN7c-ABULRubNZzXVQ'

#Notifies discord API that bot wants messages and message content from server
intents = discord.Intents.all()  # Enable all intents

#Creates bot with given command prefix
bot = commands.Bot(command_prefix='!', intents=intents)

def storeMessages(messages):
    # Connect to SQLite database (this will create the database if it doesn't exist)
    connection = sqlite3.connect("messages.db")
    cursor = connection.cursor()

    # Create a table for messages
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author TEXT,
            content TEXT
        )
    ''')

    # Insert messages into the table
    for message in messages:
        cursor.execute('''
            INSERT INTO messages (author, content) VALUES (?, ?)
        ''', (message["author"], message["content"]))

    # Commit changes and close connection
    connection.commit()
    connection.close()

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.event
async def on_message(message):
    print(f"{message.author.name}: {message.content}")
    if message.author == bot.user:
        return
    await bot.process_commands(message)

@bot.command(name='get_messages')
async def get_messages(ctx):
    # Get the channel object
    channel = ctx.channel
    
    # Retrieve all messages from the channel
    storeMessages(channel.history(limit=None))

bot.run(bot_token)
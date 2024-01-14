# INSTRUCTIONS FOR CREATING DATABASE OF MESSSAGES:
# 1. Run this script
# 2. In the channel that you want training data taken from, send !get_messages
# 3. Wait for the scipt to create the database (it will print out "Database Created")
# 4. Stop the script

import sqlite3
import discord
from discord.ext import commands
import configparser

#Gets bot_token from config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
bot_token = config['Credentials']['DISCORD_BOT_TOKEN']

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
    
    print("Database Created")

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

    # Retrieve all messages
    all_messages = await channel.history(limit=None).flatten()

    # Filter text messages based on type
    text_messages = [message for message in all_messages if message.type == discord.MessageType.default]

    storeMessages(text_messages)

bot.run(bot_token)
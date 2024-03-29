# Imitator Discord Bot

Welcome to the Imitator Discord Bot project! This Discord bot combines the power of OpenAI's GPT model and machine learning to create an interactive and dynamic chatting experience.

### Getting Started


1. **Create a Discord Bot:**
   - Create a new Discord bot on the [Discord Developer Portal](https://discord.com/developers/applications).
   - Copy the bot token.
   - Add this bot to your Discord Server. Make sure you give it permissions for reading message history and sending messages

2. **Configure OpenAI API Key:**
   - Go to [OpenAI API KEY](https://platform.openai.com/api-keys)
   - Create a new secret key
   - Copy this key down

3. **Creating a configuration file**
    - Create a file named config.ini
    - Copy-paste this text into config.ini and input your OPENAI api key and Discord Bot token keys where specified

    ```plaintext
    [Credentials]
    OPENAI_API_KEY = your-openai-api-key
    DISCORD_BOT_TOKEN = your-discord-bot-token-key
    ```

4. **Configure ML Part:**
   - Open `storeDiscordMessages.py` in the `mlBot` folder.
   - Follow the instructions to create a database of Discord messages.
   - Run `mlBotCreator.py` to create a machine learning model trained on the Discord message database.

5. **Run the Bot:**
   - Open `ImitatorBot.py`.
   - Run the script to start the AI-Powered Discord Bot.

Feel free to explore and customize the bot based on your preferences. If you have any questions or feedback, don't hesitate to reach out!

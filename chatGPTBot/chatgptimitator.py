from openai import OpenAI
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['Credentials']['OPENAI_API_KEY']


async def generate_response(person, message):
    global api_key
    aiClient = OpenAI(api_key=api_key)
    print("Generating response")
    try:
        response = aiClient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"I want you to act as {person}. I want your responses to be as close to what the actual person would say as possible. You should use your knowledge of how discord users chat to make the conversation as realistic as possible. Try to exaggerate the characteristics of the person without breaking the realism. Try to include words or phrases that the person would normally include in their speech."},
                {"role": "user", "content": message}
            ] 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
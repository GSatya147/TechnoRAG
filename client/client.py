import litellm

from configurables.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, logger

class Client:
    def __init__(self):
        pass

    def generate_response(self, messages_list: list[dict]):

        try:
            response = litellm.completion(
            api_key=DEEPSEEK_API_KEY,
            model=DEEPSEEK_MODEL,
            max_tokens=1500,
            messages=messages_list,
            temperature=0.3
            )

            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(e)
    
if __name__=="__main__":
    obj = Client()
    result = obj.generate_response(messages_list=[{"role" : "system", "content" : "You are my friend."}, {"role" : "user", "content" : "Hello"}])
    print(result)


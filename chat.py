import os
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY", "sk-5f3971844d554d73a262d6601447e32b")  
base_url = "https://api.deepseek.com"

client = OpenAI(api_key=api_key, base_url=base_url)

def get_model_layers_info(info):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system", 
                "content": "你是一个深度学习专家。我开发了一个网站，功能是深度学习模型可视化，你是我内置在该网站中的给用户提供信息的助手，简要回答用户所提的关于深度学习方面的问题。"
            },
            {
                "role": "user", 
                "content": f"用户输入：{info}"
            },
        ],
        stream=False
    )

    return response.choices[0].message.content

# 测试调用
if __name__ == "__main__":
    info = "你好，介绍一下你自己"
    
    description = get_model_layers_info(info)
    print(description)

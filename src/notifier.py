import os
import requests

def send_wechat_webhook(markdown_content: str):
    """
    发送 Markdown 内容到企业微信群机器人 Webhook
    """
    webhook_url = os.environ.get("WECHAT_WEBHOOK_URL")
    if not webhook_url:
        print("ℹ️ 未配置 WECHAT_WEBHOOK_URL 环境变量，跳过微信推送。")
        return

    print("🚀 正在推送报告到微信群...")
    payload = {
        "msgtype": "markdown",
        "markdown": {
            "content": markdown_content
        }
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ 微信推送成功！")
        else:
            print(f"❌ 微信推送失败，状态码: {response.status_code}, 返回: {response.text}")
    except Exception as e:
        print(f"❌ 微信推送出错: {e}")

import os
import requests

def send_wechat_webhook(markdown_content: str):
    """
    发送 Markdown 内容到企业微信群机器人 Webhook
    自动处理 4096 字节长度限制（超出则分多条发送）
    """
    webhook_url = os.environ.get("WECHAT_WEBHOOK_URL")
    if not webhook_url:
        print("ℹ️ 未配置 WECHAT_WEBHOOK_URL 环境变量，跳过微信推送。")
        print("   如果你在 GitHub Actions 运行，请确保在仓库 Settings -> Secrets 中添加了 WECHAT_WEBHOOK_URL。")
        return

    print("🚀 正在推送报告到微信群...")
    
    # 微信 webhook markdown 有 4096 bytes 的限制
    # 将文本按行分割，组装成多个 payload，每个不超过 4000 字节
    lines = markdown_content.split('\n')
    chunks = []
    current_chunk = ""
    
    for line in lines:
        # 如果单行加入后超过 4000 字节，则将当前 chunk 保存，开启新 chunk
        if len((current_chunk + line + "\n").encode('utf-8')) > 4000:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
            
    if current_chunk:
        chunks.append(current_chunk)

    for i, chunk in enumerate(chunks):
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "content": chunk
            }
        }
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"✅ 微信推送成功！(分段 {i+1}/{len(chunks)})")
            else:
                print(f"❌ 微信推送失败，状态码: {response.status_code}, 返回: {response.text}")
        except Exception as e:
            print(f"❌ 微信推送出错: {e}")

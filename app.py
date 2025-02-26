import urllib.parse
import requests
import gradio as gr
from transformers import pipeline, MBart50Tokenizer

# 🟢 API Templates للحصول على بيانات ويكيبيديا
SEARCH_TEMPLATE = "https://ar.wikipedia.org/w/api.php?action=opensearch&search=%s&limit=1&namespace=0&format=json"
CONTENT_TEMPLATE = "https://ar.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles=%s"

# 🔥 استخدام tokenizer الصحيح
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", use_fast=False)
summarizer = pipeline("summarization", model="facebook/mbart-large-50", tokenizer=tokenizer)

def search_wikipedia(query):
    """ البحث في ويكيبيديا العربية وإرجاع ملخص من المقال الأول. """
    query_encoded = urllib.parse.quote_plus(query)
    search_response = requests.get(SEARCH_TEMPLATE % query_encoded).json()

    if not search_response or not search_response[1]:  
        return "❌ لم يتم العثور على نتائج.", ""

    # 🟢 جلب أول نتيجة
    page_title = search_response[1][0]
    page_encoded = urllib.parse.quote_plus(page_title)
    
    # 🟢 جلب محتوى المقالة
    content_response = requests.get(CONTENT_TEMPLATE % page_encoded).json()
    pages = content_response.get("query", {}).get("pages", {})
    
    if not pages:
        return "❌ لم يتم العثور على المحتوى.", ""

    content = list(pages.values())[0].get("extract", "")

    if not content:
        return "❌ المقالة لا تحتوي على معلومات كافية.", ""

    # 🟢 ضبط طول الإدخال للتوافق مع حدود النموذج
    max_input_length = 1024
    content = content[:max_input_length]

    # 🟢 تحسين التلخيص بناءً على طول المقالة
    max_length = 200 if len(content) > 1000 else 100
    min_length = 50 if len(content) > 500 else 30

    summary = summarizer(content, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    source = search_response[3][0]  # رابط المقال الأصلي
    return summary, source

def chatbot_response(message, history):
    """ دالة التفاعل مع المستخدم """
    summary, source = search_wikipedia(message)
    response = f"🔹 **ملخص ويكيبيديا:**\n\n{summary}"
    if source:
        response += f"\n\n🔗 **المصدر:** [اضغط هنا]({source})"
    history.append((message, response))
    return history

# 🔥 واجهة Gradio المحسنة
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 بوت ويكيبيديا العربية")
    gr.Markdown("🔹 هذا البوت يستخدم ويكيبيديا العربية للبحث عن المعلومات وإعطاء ملخص عنها.")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="🔍 اكتب سؤالك هنا:")
    clear = gr.Button("🧩 مسح المحادثة")

    msg.submit(chatbot_response, [msg, chatbot], chatbot).then(
        lambda _: "", None, [msg], queue=False  # 🟢 تصحيح التحديث التلقائي لحقل الإدخال
    )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)  # 🟢 تمكين المشاركة العامة

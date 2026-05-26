import streamlit as st
import whisper
import tempfile
import os
import ssl

# 處理 Mac 的 SSL 憑證問題
ssl._create_default_https_context = ssl._create_unverified_context


def format_time(seconds):
    """將秒數轉換為 SRT 字幕的時間格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ================= 網頁前端介面設計 =================

# 設定網頁標題與圖示
st.set_page_config(page_title="AI 語音逐字稿產生器")
st.title("AI 語音逐字稿產生器")
st.write("上傳你的影音檔案（支援 m4a, mp3, mp4 等），讓 AI 自動為你生成帶有時間軸的 SRT 字幕檔！")

# 1. 建立檔案上傳元件
uploaded_file = st.file_uploader("請選擇影音檔案...", type=["m4a", "mp3", "wav", "mp4", "mov"])

if uploaded_file is not None:
    st.info(f"成功偵測到檔案：{uploaded_file.name} (大小: {uploaded_file.size / 1024 / 1024:.2f} MB)")

    # 2. 建立觸發按鈕
    if st.button("開始執行 AI 辨識"):

        # 顯示網頁載入動畫
        with st.spinner("AI 正在努力聆聽並轉錄中，請稍候... (這可能需要幾分鐘)"):
            try:
                # 核心技巧：將 Streamlit 記憶體中的檔案寫入暫存檔
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name  # 取得暫存檔的真實路徑

                # 載入 Whisper 模型
                model = whisper.load_model("base")

                # 進行辨識 (關閉 fp16 以配合 CPU 運算)
                result = model.transcribe(temp_file_path, fp16=False)

                # 辨識完成後，將暫存檔從硬碟刪除，保持電腦整潔
                os.unlink(temp_file_path)

                # 3. 處理並組裝 SRT 字幕字串
                srt_content = ""
                for i, segment in enumerate(result["segments"]):
                    start_time = format_time(segment["start"])
                    end_time = format_time(segment["end"])
                    text = segment["text"].strip()

                    srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n"

                st.success(" 逐字稿辨識完成！")

                # 4. 提供下載按鈕，讓使用者下載產出的 SRT 檔案
                st.download_button(
                    label="下載 SRT 字幕檔案",
                    data=srt_content,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}.srt",
                    mime="text/plain"
                )

                # 在網頁上即時預覽前幾段內容
                st.subheader("逐字稿內容預覽")
                st.text_area("前段內容：", value=srt_content[:1000] + "\n... (以下省略)", height=300)

            except Exception as e:
                st.error(f"程式執行發生錯誤：{e}")
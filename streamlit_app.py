
# save as invoice_app.py
import streamlit as st
import keras_ocr, re, numpy as np

pipeline = keras_ocr.pipeline.Pipeline()

def scan_invoice(path):
    image = keras_ocr.tools.read(path)
    prediction = pipeline.recognize([image])[0]
    tokens = [t for t, _ in prediction]
    all_text = " ".join(tokens)
    
    result = {"invoice_no": None, "date": None, "total": None}
    if m := re.search(r'Invoice\s*No[:\s]*([A-Za-z0-9/-]+)', all_text, re.IGNORECASE):
        result["invoice_no"] = m.group(1)
    if m := re.search(r'Dated[:\s]*([0-9]{1,2}[-./][A-Za-z]{3}[-./][0-9]{2,4})', all_text, re.IGNORECASE):
        result["date"] = m.group(1)
    if m := re.search(r'IGST.*?([0-9.,]+)', all_text, re.IGNORECASE):
        result["igst"] = m.group(1)
    if m := re.search(r'Amount\s*([0-9.,]+)', all_text, re.IGNORECASE):
        result["total"] = m.group(1)
    
    return result, prediction

# Streamlit UI
st.title("📄 Quick Invoice Scanner")

uploaded = st.file_uploader("Upload an Invoice", type=["jpg","png"])
if uploaded:
    with open("temp.png", "wb") as f:
        f.write(uploaded.read())
    data, prediction = scan_invoice("temp.png")
    
    st.subheader("Extracted Fields")
    st.json(data)
    
    st.subheader("OCR Preview")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    image = keras_ocr.tools.read("temp.png")
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=ax)
    st.pyplot(fig)

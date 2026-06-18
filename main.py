from io import BytesIO
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st


def extract_table_from_image(image_bytes):
    """Processes the image from raw bytes and returns a list of lists containing cell data."""
    # Convert uploaded file bytes into an OpenCV image
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode the image file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding the image to get a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
        1
    ]

    # Detect horizontal and vertical lines to find table structure
    cols = thresh.shape[1]
    rows = thresh.shape[0]
    horizontal_size = cols // 40
    vertical_size = rows // 40

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1)
    )
    detect_horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_size)
    )
    detect_vertical = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    table_mask = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0)
    table_mask = cv2.threshold(table_mask, 0, 255, cv2.THRESH_BINARY)[1]

    # Find contours of the cells
    contours, _ = cv2.findContours(
        table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    box_list = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (
            w > 20
            and h > 15
            and w < (img.shape[1] * 0.95)
            and h < (img.shape[0] * 0.95)
        ):
            box_list.append((x, y, w, h))

    if not box_list:
        raise ValueError(
            "No clear table structure or grid lines detected in the image."
        )

    # Group cell coordinates into rows based on 'y' position
    box_list = sorted(box_list, key=lambda b: (b[1], b[0]))

    rows_data = []
    current_row = []
    y_tolerance = 10
    prev_y = box_list[0][1]

    for box in box_list:
        x, y, w, h = box
        if y - prev_y > y_tolerance:
            current_row = sorted(current_row, key=lambda b: b[0])
            rows_data.append(current_row)
            current_row = []
            prev_y = y
        current_row.append(box)

    if current_row:
        current_row = sorted(current_row, key=lambda b: b[0])
        rows_data.append(current_row)

    # OCR text extraction from each isolated cell
    table_content = []
    config = "--psm 6"

    for row in rows_data:
        row_cells = []
        for box in row:
            x, y, w, h = box
            padding = 2
            cell_crop = gray[
                max(0, y - padding) : min(img.shape[0], y + h + padding),
                max(0, x - padding) : min(img.shape[1], x + w + padding),
            ]

            text = pytesseract.image_to_string(cell_crop, config=config).strip()
            row_cells.append(text)
        table_content.append(row_cells)

    # Normalize row lengths
    max_cols = max(len(r) for r in table_content)
    for r in table_content:
        while len(r) < max_cols:
            r.append("")

    return table_content


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Generic Image Table Extractor", layout="wide")
st.title("📊 Generic Image Table Extractor")
st.write(
    "Upload any image containing a structured data table to convert it into a downloadable Excel sheet."
)

uploaded_file = st.file_uploader(
    "Choose a table image...", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Process and Extract Table"):
        with st.spinner("Extracting cells and reading text via OCR..."):
            try:
                # Read file raw bytes
                img_bytes = uploaded_file.read()
                raw_data = extract_table_from_image(img_bytes)

                # Format into DataFrame
                df = pd.DataFrame(raw_data)
                if len(df) > 1:
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)

                st.success("Extraction complete! Preview your table below:")
                st.dataframe(df, use_container_width=True)

                # Convert DataFrame to an Excel file entirely in-memory
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Extracted Table")
                excel_data = excel_buffer.getvalue()

                # Streamlit Download Button
                st.download_button(
                    label="💾 Download Table as Excel File",
                    data=excel_data,
                    file_name=f"extracted_{uploaded_file.name.split('.')[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

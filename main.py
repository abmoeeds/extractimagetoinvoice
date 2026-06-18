from io import BytesIO
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st


def extract_table_from_image(image_bytes):
    """Processes the image from raw bytes and returns a list of lists containing cell data."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode the image file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding handles shadows/lighting variations better dynamically
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    cols = thresh.shape[1]
    rows = thresh.shape[0]

    # Making scale more sensitive to capture thinner or fainter grid lines
    horizontal_size = cols // 50
    vertical_size = rows // 50

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

    contours, _ = cv2.findContours(
        table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    box_list = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Drop extreme noise but keep smaller valid cells
        if (
            w > 15
            and h > 10
            and w < (img.shape[1] * 0.98)
            and h < (img.shape[0] * 0.98)
        ):
            box_list.append((x, y, w, h))

    if not box_list:
        raise ValueError(
            "No clear table structure or grid lines detected in the image."
        )

    box_list = sorted(box_list, key=lambda b: (b[1], b[0]))

    rows_data = []
    current_row = []
    y_tolerance = 12  # Slightly widened tolerance for rows that slant slightly
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

    table_content = []
    config = "--psm 6"

    for row in rows_data:
        row_cells = []
        for box in row:
            x, y, w, h = box
            padding = 1
            cell_crop = gray[
                max(0, y - padding) : min(img.shape[0], y + h + padding),
                max(0, x - padding) : min(img.shape[1], x + w + padding),
            ]

            text = pytesseract.image_to_string(cell_crop, config=config).strip()
            row_cells.append(text)
        table_content.append(row_cells)

    max_cols = max(len(r) for r in table_content)
    for r in table_content:
        while len(r) < max_cols:
            r.append("")

    return table_content


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Generic Image Table Extractor", layout="wide")
st.title("📊 Generic Image Table Extractor")
st.write("Upload any image containing a structured data table.")

uploaded_file = st.file_uploader(
    "Choose a table image...", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Process and Extract Table"):
        with st.spinner("Extracting cells and reading text via OCR..."):
            try:
                img_bytes = uploaded_file.read()
                raw_data = extract_table_from_image(img_bytes)

                df = pd.DataFrame(raw_data)

                # --- FIX FOR DUPLICATE/EMPTY HEADERS ---
                if len(df) > 0:
                    header_row = df.iloc[0].astype(str).tolist()

                    # Clean headers and handle blanks/duplicates safely
                    seen = {}
                    new_headers = []
                    for i, head in enumerate(header_row):
                        head_clean = head.strip().replace("\n", " ")
                        if not head_clean:
                            head_clean = f"Column_{i+1}"

                        if head_clean in seen:
                            seen[head_clean] += 1
                            head_clean = f"{head_clean}_{seen[head_clean]}"
                        else:
                            seen[head_clean] = 0

                        new_headers.append(head_clean)

                    df.columns = new_headers
                    df = df[1:].reset_index(drop=True)

                st.success("Extraction complete!")
                st.dataframe(df, use_container_width=True)

                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Extracted Table")
                excel_data = excel_buffer.getvalue()

                st.download_button(
                    label="💾 Download Table as Excel File",
                    data=excel_data,
                    file_name=f"extracted_{uploaded_file.name.split('.')[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

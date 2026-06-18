from io import BytesIO
import pandas as pd
import streamlit as st
from img2table.document import Image
from img2table.ocr import TesseractOCR

# --- STREAMLIT UI SETUP ---
st.set_page_config(
    page_title="Smart Image Table Extractor", layout="wide", page_icon="📊"
)
st.title("📊 Smart Image Table Extractor")
st.write(
    "Upload any image containing a multi-column table to structuralize and export it perfectly to Excel."
)

uploaded_file = st.file_uploader(
    "Choose a table image...", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    # Safely present the original document image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Process and Structuralize Table"):
        with st.spinner("Analyzing layout and reading data cleanly via OCR..."):
            try:
                # 1. Read file bytes directly into memory
                img_bytes = uploaded_file.getvalue()

                # 2. Instantiate modern img2table Document & OCR Engine wrapper
                # It natively runs inside Streamlit Cloud's Linux environment
                doc = Image(src=img_bytes)
                ocr = TesseractOCR(lang="eng")

                # 3. Extract tables natively factoring implicit row structural analysis
                extracted_tables = doc.extract_tables(
                    ocr=ocr, implicit_rows=True, borderless_tables=False
                )

                if not extracted_tables:
                    st.error(
                        "No structured table layout could be automatically isolated from this image. Ensure the image has visible columns."
                    )
                else:
                    st.success(
                        f"Successfully found and reconstructed {len(extracted_tables)} table(s)!"
                    )

                    # Create an in-memory byte buffer for writing the Excel file
                    excel_buffer = BytesIO()

                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        for idx, table in enumerate(extracted_tables):
                            # Convert native img2table struct directly to a Pandas DataFrame
                            df = table.df

                            # Clean up row index headers if first row acts as the primary column title
                            if len(df) > 1:
                                df.columns = df.iloc[0].astype(str)
                                df = df[1:].reset_index(drop=True)

                            # Sanitize empty naming or line breaks out of headers
                            df.columns = [
                                str(c).replace("\n", " ").strip()
                                if str(c).strip()
                                else f"Column_{i+1}"
                                for i, c in enumerate(df.columns)
                            ]

                            # Display the extracted preview grid visually on webpage
                            st.subheader(f"Table Preview {idx + 1}")
                            st.dataframe(df, use_container_width=True)

                            # Append sheet structure to data buffer
                            df.to_excel(
                                writer,
                                index=False,
                                sheet_name=f"Table_{idx + 1}",
                            )

                    # Fetch raw binary representation of completed workbook
                    excel_data = excel_buffer.getvalue()

                    # Provide download button for the spreadsheet
                    st.download_button(
                        label="💾 Download Structured Excel Workbook",
                        data=excel_data,
                        file_name=f"extracted_{uploaded_file.name.split('.')[0]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            except Exception as e:
                st.error(
                    f"An error occurred during layout conversion: {str(e)}"
                )

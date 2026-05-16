import os
import psycopg2
import streamlit as st
from PIL import Image, ImageDraw
import io

# --- Database Connection ---
DSN = os.environ.get("FACE_SCANNER_DSN")
if not DSN:
    st.error("FACE_SCANNER_DSN environment variable not set.")
    st.stop()

def get_connection():
    return psycopg2.connect(DSN)

# --- Data Fetching ---
def fetch_persons():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, cluster_label, face_count
                FROM persons
                ORDER BY COALESCE(name, ''), id
            """)
            return cur.fetchall()

def fetch_faces_for_person(person_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT fd.id, p.file_path, fd.bounding_box
                FROM face_detections fd
                JOIN photos p ON fd.photo_id = p.id
                WHERE fd.person_id = %s
                ORDER BY p.file_path, fd.id
            """, (person_id,))
            return cur.fetchall()

# --- Image Utilities ---
def load_and_crop_face(image_path, bounding_box):
    try:
        img = Image.open(image_path)
        if bounding_box:
            # bounding_box is a dict with normalized coordinates
            x = bounding_box.get('x')
            y = bounding_box.get('y')
            w = bounding_box.get('width')
            h = bounding_box.get('height')
            if None not in (x, y, w, h):
                img_w, img_h = img.size
                left = int(x * img_w)
                top = int(y * img_h)
                right = int((x + w) * img_w)
                bottom = int((y + h) * img_h)
                face_img = img.crop((left, top, right, bottom))
                return face_img
        return img
    except Exception as e:
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Face Clusters Viewer", layout="wide")
st.title("Face Clusters Viewer")

persons = fetch_persons()
person_options = [
    f"{p[1] if p[1] else 'Unnamed'} (ID: {p[0]}) [Faces: {p[3]}]" for p in persons
]
person_id_map = {person_options[i]: persons[i][0] for i in range(len(persons))}

selected = st.sidebar.selectbox("Select a person/cluster:", person_options)
selected_person_id = person_id_map[selected]

faces = fetch_faces_for_person(selected_person_id)

st.header(f"Faces for: {selected}")

cols = st.columns(5)
for idx, (face_id, file_path, bounding_box) in enumerate(faces):
    # bounding_box is JSONB, so it's a dict already in psycopg2
    img = load_and_crop_face(file_path, bounding_box)
    if img:
        with cols[idx % 5]:
            st.image(img, caption=os.path.basename(file_path), use_column_width=True)
    else:
        with cols[idx % 5]:
            st.write(f"Could not load: {file_path}")

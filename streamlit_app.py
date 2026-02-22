import streamlit as st
import requests
import time

API_BASE = "http://localhost:8000"

st.set_page_config(layout="wide")

# =====================================================
# CLEAN SAAS CSS (Minimal + Hover Delete)
# =====================================================

st.markdown("""
<style>

section[data-testid="stSidebar"] {
    background-color: #F9FAFB;
}

/* Document row */
.doc-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    border-radius: 6px;
    margin-bottom: 6px;
    transition: background 0.2s ease;
}

.doc-row:hover {
    background-color: #F3F4F6;
}

/* Status borders */
.status-ready {
    border-left: 4px solid #22C55E;
}

.status-processing {
    border-left: 4px solid #F59E0B;
}

.status-failed {
    border-left: 4px solid #EF4444;
}

/* Filename */
.file-name {
    font-size: 14px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
}

/* Hide delete button by default */
div[data-testid="stButton"] button {
    opacity: 0;
    transition: opacity 0.2s ease;
}

/* Show delete on hover */
div[data-testid="stHorizontalBlock"]:hover div[data-testid="stButton"] button {
    opacity: 1;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================

if "message" not in st.session_state:
    st.session_state.message = None

if "message_type" not in st.session_state:
    st.session_state.message_type = None

# =====================================================
# HELPERS
# =====================================================

def fetch_workspaces():
    try:
        r = requests.get(f"{API_BASE}/workspaces")
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []

def fetch_documents(workspace_id):
    try:
        r = requests.get(
            f"{API_BASE}/workspaces/{workspace_id}/documents"
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []

def fetch_progress(workspace_id, doc_id):
    try:
        r = requests.get(
            f"{API_BASE}/workspaces/{workspace_id}/documents/{doc_id}/progress"
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def status_class(status):
    return {
        "READY": "status-ready",
        "PROCESSING": "status-processing",
        "FAILED": "status-failed"
    }.get(status, "")

# =====================================================
# SIDEBAR
# =====================================================

workspace_list = fetch_workspaces()
selected_workspace = None

if workspace_list:
    selected_workspace = st.sidebar.selectbox(
        "Workspace",
        workspace_list
    )

    # ---- Delete Workspace Button ----
    st.sidebar.markdown(
        """
        <div style='margin-top:6px;'>
            <small style='color:#9CA3AF;'>Danger Zone</small>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.sidebar.button(
        "🗑 Delete Workspace",
        use_container_width=True
    ):
        res = requests.delete(
            f"{API_BASE}/workspaces/{selected_workspace}"
        )

        if res.status_code == 200:
            st.session_state.message = "Workspace deleted"
            st.session_state.message_type = "success"
        else:
            st.session_state.message = "Failed to delete workspace"
            st.session_state.message_type = "error"

        st.rerun()

else:
    st.sidebar.info("No workspaces yet")

# ---------- Create Workspace ----------

st.sidebar.markdown("### New Workspace")

new_workspace = st.sidebar.text_input(
    "Workspace Name",
    placeholder="Enter workspace name",
    label_visibility="collapsed"
)

if st.sidebar.button("Create Workspace"):
    if new_workspace.strip():
        res = requests.post(
            f"{API_BASE}/workspaces/{new_workspace}"
        )
        if res.status_code == 200:
            st.session_state.message = "Workspace created"
            st.session_state.message_type = "success"
        else:
            st.session_state.message = "Workspace already exists"
            st.session_state.message_type = "error"
    else:
        st.session_state.message = "Workspace name cannot be empty"
        st.session_state.message_type = "warning"

    st.rerun()

# =====================================================
# DOCUMENT LIST (Inline Hover Delete)
# =====================================================

if selected_workspace:

    docs = fetch_documents(selected_workspace)

    st.sidebar.markdown("### Documents")

    if not docs:
        st.sidebar.caption("No documents uploaded")
    else:
        for doc in docs:

            row_class = status_class(doc["status"])

            col1, col2 = st.sidebar.columns([9, 1])

            with col1:
                st.markdown(
                    f"""
                    <div class="doc-row {row_class}">
                        <div class="file-name" title="{doc['file_name']}">
                            {doc['file_name']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Progress only if processing
                if doc["status"] == "PROCESSING":
                    progress_data = fetch_progress(
                        selected_workspace,
                        doc["id"]
                    )

                    if progress_data:
                        percent = progress_data.get("percentage", 0)
                        st.progress(percent / 100)

            with col2:
                if st.button(
                    "🗑",
                    key=f"delete_{doc['id']}",
                    help="Delete document"
                ):
                    res = requests.delete(
                        f"{API_BASE}/workspaces/{selected_workspace}/documents/{doc['id']}"
                    )

                    if res.status_code == 200:
                        st.session_state.message = "Document deleted"
                        st.session_state.message_type = "success"
                    else:
                        st.session_state.message = "Failed to delete document"
                        st.session_state.message_type = "error"

                    st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("Delete Workspace"):
        res = requests.delete(
            f"{API_BASE}/workspaces/{selected_workspace}"
        )

        if res.status_code == 200:
            st.session_state.message = "Workspace deleted"
            st.session_state.message_type = "success"
        else:
            st.session_state.message = "Failed to delete workspace"
            st.session_state.message_type = "error"

        st.rerun()

# =====================================================
# MAIN AREA
# =====================================================

st.title("Document Manager")

if st.session_state.message:
    if st.session_state.message_type == "success":
        st.success(st.session_state.message)
    elif st.session_state.message_type == "error":
        st.error(st.session_state.message)
    elif st.session_state.message_type == "warning":
        st.warning(st.session_state.message)

    st.session_state.message = None
    st.session_state.message_type = None

if not selected_workspace:
    st.info("Create or select a workspace to begin.")
    st.stop()

st.subheader(f"Workspace: {selected_workspace}")

# =====================================================
# FILE UPLOAD
# =====================================================

st.markdown("### Upload Documents")

uploaded_files = st.file_uploader(
    "Upload files",
    accept_multiple_files=True,
    help="Supported: PDF, DOCX, JSON, XML, TXT, Images, ZIP"
)

if uploaded_files:
    if st.button("Upload"):
        progress = st.progress(0)
        total = len(uploaded_files)

        for index, file in enumerate(uploaded_files):

            files = {"file": (file.name, file.getvalue())}

            response = requests.post(
                f"{API_BASE}/workspaces/{selected_workspace}/upload",
                files=files
            )

            if response.status_code == 200:
                st.session_state.message = f"{file.name} uploaded"
                st.session_state.message_type = "success"
            elif response.status_code == 409:
                st.session_state.message = f"{file.name} duplicate"
                st.session_state.message_type = "warning"
            else:
                st.session_state.message = f"{file.name} failed"
                st.session_state.message_type = "error"

            progress.progress((index + 1) / total)

        time.sleep(0.5)
        st.rerun()

# =====================================================
# AUTO REFRESH
# =====================================================

if any(doc["status"] == "PROCESSING" for doc in docs):
    time.sleep(2)
    st.rerun()
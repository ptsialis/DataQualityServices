import streamlit as st
import global_controller as gc

st.header("Summarising Page", divider="rainbow")
st.write(st.session_state.Encoded_Data_Frame)

metadata_map = gc.get_all_metadata()
for key in metadata_map.keys():
    st.subheader(key,divider="rainbow")
    st.write(metadata_map[key])
#st.write(gc.get_all_metadata())
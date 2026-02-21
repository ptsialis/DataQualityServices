import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps

def successful_labelling_view():
        # if not st.session_state.dataset_labelling_done:
        #         with st.spinner("Labelling dataset with the selected model..."):
        #                 # Only run once
        #                 result = st.session_state.controller.process_dataset_labelling(
        #                         st.session_state.model_for_dataset,
        #                         st.session_state.dataset_info
        #                 )

        #                 st.success("The dataset has been successfully labelled.")

        #                 # Save results to session state
        #                 st.session_state.output_zip_path = result["zip_path"]
        #                 st.session_state.stats = result["stats"]
        #                 st.session_state.preview = result["preview"]
        #                 st.session_state.dataset_labelling_done = True
                        
        if "dataset_labelling_done" not in st.session_state:
                st.session_state.dataset_labelling_done = False

        # Case 1: Results already in session state (manual labelling path)
        if not st.session_state.dataset_labelling_done and "stats" in st.session_state:
                st.success("✅ The dataset has been successfully labelled.")
                st.session_state.dataset_labelling_done = True

        # Case 2: If not manual, fall back to model-based labelling (default path)
        elif not st.session_state.dataset_labelling_done:
                with st.spinner("Labelling dataset with the selected model..."):
                        result = st.session_state.controller.process_dataset_labelling(
                                st.session_state.model_for_dataset,
                                st.session_state.dataset_info
                        )
                        st.session_state.output_zip_path = result["zip_path"]
                        st.session_state.stats = result["stats"]
                        st.session_state.preview = result["preview"]
                        st.success("✅ The dataset has been successfully labelled.")
                        st.session_state.dataset_labelling_done = True

        # Plotting 
        stats = st.session_state.stats
        st.subheader("📊 Labelling Statistics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Images", stats["total_images"])
        col2.metric("Number of Classes", stats["num_classes"])
        col3.metric("ZIP Size (MB)", stats["zip_file_size_MB"])
        
        # Class distribution plot
        st.subheader("Label Distribution")

        # Matplotlib 
        # import matplotlib.pyplot as plt
        # labels = list(stats["class_distribution"].keys())
        # counts = list(stats["class_distribution"].values())

        # fig, ax = plt.subplots()
        # ax.bar(labels, counts)
        # ax.set_xlabel("Class Label")
        # ax.set_ylabel("Image Count")
        # ax.set_title("Label Distribution")
        # st.pyplot(fig)
        
        
        # Streamlit Charts
        import altair as alt
        
        class_dist_df = pd.DataFrame([
                {"class_label": str(label), "image_count": count}
                for label, count in stats["class_distribution"].items()
                ])

        # Create an interactive bar chart with colour
        bar_chart = alt.Chart(class_dist_df).mark_bar().encode(
                x=alt.X("class_label:N", title="Class Label"),
                y=alt.Y("image_count:Q", title="Image Count"),
                color=alt.Color("class_label:N", legend=None),
                tooltip=["class_label", "image_count"]
        ).properties(
                width=600,
                height=400,
                title= "Interactive Bar Chart Showing the Label Distribution"
        )

        st.altair_chart(bar_chart, use_container_width=True)

        # Preview
        st.subheader("🔎 Preview Labelled Dataset")

        preview_items = list(st.session_state.preview.items())

        max_preview_size = 5
        # Show first 5 classes directly (expanded)
        for label, images in preview_items[:max_preview_size]:
                with st.expander(f"Class {label} (showing {len(images)} samples)", expanded=True):
                        cols = st.columns(5)
                        for i, img_tensor in enumerate(images):
                                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                                img_pil = Image.fromarray((img_np * 255).astype("uint8"))
                                cols[i % 5].image(img_pil, caption=f"Sample {i + 1}", use_container_width=True)

        # Add toggle for remaining classes
        if len(preview_items) > max_preview_size:
                show_more = st.checkbox(f"Show remaining {len(preview_items) - max_preview_size} classes")
                if show_more:
                        for label, images in preview_items[max_preview_size:]:
                                with st.expander(f"Class {label} (showing {len(images)} samples)"):
                                        cols = st.columns(5)
                                        for i, img_tensor in enumerate(images):
                                                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                                                img_pil = Image.fromarray((img_np * 255).astype("uint8"))
                                                cols[i % 5].image(img_pil, caption=f"Sample {i + 1}", use_container_width=True)

        if st.button("Proceed to \"Download\""):
                st.session_state.page = "download"
                st.rerun()



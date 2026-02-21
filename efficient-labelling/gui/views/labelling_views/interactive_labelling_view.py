import streamlit as st

def interactive_labelling_view():
    st.title("Interactive Labelling")

    dataset_controller = st.session_state.dataset_controller

    # Phase 1: User specifies the maximum number of unique classes allowed.
    st.header("Labelling Session Setup")
    max_classes = st.number_input(
        "Enter the maximum number of unique classes allowed", 
        min_value=1, value=3, step=1
    )

    # Ensure dataset_controller exists in session state
    if "dataset_controller" not in st.session_state:
        st.error("DatasetController not found in session state. Please upload a dataset first.")
        return

    dataset_controller = st.session_state.dataset_controller

    # Initialize session state for data subset and labels
    if "data_subset" not in st.session_state:
        st.session_state.data_subset = []
    if "labels_subset" not in st.session_state:
        st.session_state.labels_subset = {}

    # Phase 2: Fetch a subset of images (only if not already loaded)
    if st.button("Load Data"):
        with st.spinner("Fetching data from server..."):
            st.session_state.data_subset = dataset_controller.get_labelling_subset(num_subset=5)
            st.session_state.labels_subset = {item['id']: "" for item in st.session_state.data_subset}
            st.success("Data loaded successfully!")

    # Phase 3: Display images and corresponding label input fields.
    if st.session_state.data_subset:
        st.header("Label the Images")

        for item in st.session_state.data_subset:
            st.subheader(f"Image {item['id']}")
            st.image(item['file'], caption=f"Image {item['id']}")  # Display actual image
            
            # Persist label input in session state
            label_input = st.text_input(
                f"Enter label for Image {item['id']}", 
                value=st.session_state.labels_subset.get(item['id'], ""),
                key=f"label_{item['id']}"
            )

            # Update session state dynamically when user types
            st.session_state.labels_subset[item['id']] = label_input

        # Phase 4: Validate and submit the labels.
        if st.button("Submit Labels"):
            # Clean up the labels by stripping whitespace and excluding empties.
            labels = [label.strip() for label in st.session_state.labels_subset.values() if label.strip()]
            unique_labels = set(labels)
            
            # TODO: Code that submits it to the server and handles the response. 

            if len(unique_labels) > max_classes:
                st.error(
                    f"Error: You have entered {len(unique_labels)} unique classes, "
                    f"which exceeds the maximum allowed ({max_classes}). Please refine your labels."
                )
            else:
                st.session_state.subset_labelled = dataset_controller.validate_subset_labelling(None)  
                st.session_state.page = "processing"
                
                import time
                time.sleep(1)
                st.rerun()
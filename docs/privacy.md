# Privacy and data handling

- Uploaded or camera-captured image bytes are written to a temporary local directory only while inference runs. The temporary directory is removed when processing exits.
- The local SQLite database stores an inspection timestamp, a generic source label (uploaded_image.jpg or camera_capture.jpg), decision, predicted class, and confidence. It does not store image bytes or the user's original filename.
- Local database files are excluded from Git and created in the operating-system user data directory. On POSIX systems, the application attempts restrictive directory/file permissions.
- The application performs inference locally and does not call an external prediction API.
- The launcher binds Streamlit to loopback (127.0.0.1) and disables Streamlit usage statistics.
- This description applies to this repository's code. Deployments, reverse proxies, hosting providers, browsers, and operating systems may have separate logging or telemetry behaviour.
- Do not upload customer images, production data, personal information, credentials, or confidential company information to public issues or repositories.

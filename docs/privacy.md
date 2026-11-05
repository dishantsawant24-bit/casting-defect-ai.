# Privacy and data handling

- Uploaded/camera image bytes are written to a temporary local directory only while inference runs. The temporary directory is removed when processing exits.
- The app stores inspection metadata in a local SQLite database: timestamp, sanitized image filename, decision, predicted class, and confidence. Image bytes are not stored in the database.
- Local database files are excluded from Git and created in the operating-system user data directory. On POSIX systems, the application attempts restrictive directory/file permissions.
- The application performs inference locally and does not call an external prediction API.
- The launcher binds Streamlit to loopback (`127.0.0.1`) and disables Streamlit usage statistics.
- This description applies to the code in this repository. Deployments, reverse proxies, hosting providers, browsers, and operating systems may have separate logging or telemetry behaviour.
- Do not upload customer images, production data, personal information, credentials, or confidential company information to public issues or repositories.

1. **Navigate to the `100-docts-rag` directory** and run the following command to start services using Docker Compose:

   ```bash
   docker-compose up -d
   ```

2. **Build and run the Docker image** to create and start the container.

3. **Navigate to the `backend` directory** and start the FastAPI server using:

   ```bash
   uvicorn app.main:app --reload
   ```

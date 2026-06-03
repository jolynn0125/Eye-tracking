# Video Attention App

React + Vite app with two local videos, timed overlay prompts, and a MongoDB-backed session API.

## Setup

1. Open the project folder:

	 ```bash
	 cd frontend-repo/video-attention-app
	 ```

2. Make sure `.env.local` exists with your MongoDB connection string:

	 ```bash
	 MONGODB_URI=your-mongodb-connection-string
	 API_PORT=4000
	 ```

3. Install dependencies with Yarn:

	 ```bash
	 yarn
	 ```

4. Start the app:

	 ```bash
	 yarn dev
	 ```

	 This starts both the Vite frontend and the Express API.

## Local Videos

Place the two mp4 files in:

- `public/videos/video1.mp4`
- `public/videos/video2.mp4`

The React code references them as `/videos/video1.mp4` and `/videos/video2.mp4`.

## Behavior

- The app opens with a modal that requires name and email.
- The user cannot enter the video viewer until the form is submitted successfully.
- Refreshing the page resets the session and shows the modal again.
- Two tab buttons switch between the videos.
- Each video keeps its own overlay timing and click logging.
- Every overlay click is sent to the API and stored in MongoDB.

## MongoDB Shape

The API stores users in the `affective_computing` database and the `users` collection.

Documents are shaped like:

```json
{
	"_id": "session-id",
	"name": "User Name",
	"email": "user@example.com",
	"video1": {
		"promptAttempts": 0
	},
	"video2": {
		"promptAttempts": 0
	}
}
```

When the overlay button is clicked, the matching `promptAttempts` field increments.

## Useful Scripts

- `yarn dev` - run frontend and API together
- `yarn build` - build the frontend
- `yarn lint` - run ESLint

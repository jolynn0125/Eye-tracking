import express from 'express'
import cors from 'cors'
import { MongoClient } from 'mongodb'
import { config as loadEnv } from 'dotenv'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'

const envPath = resolve(process.cwd(), '.env.local')

if (existsSync(envPath)) {
  loadEnv({ path: envPath })
} else {
  loadEnv()
}

const mongoUri = process.env.MONGODB_URI
const port = Number(process.env.PORT ?? process.env.API_PORT ?? 4000)
const databaseName = 'affective_computing'
const collectionName = 'users'
const clientBuildPath = resolve(process.cwd(), 'dist')

if (!mongoUri) {
  throw new Error('MONGODB_URI is required')
}

const client = new MongoClient(mongoUri)
const app = express()

app.use(cors())
app.use(express.json())

let usersCollection

async function getUsersCollection() {
  if (usersCollection) {
    return usersCollection
  }

  await client.connect()
  usersCollection = client.db(databaseName).collection(collectionName)
  return usersCollection
}

function buildUserDocument({ sessionId, name, email }) {
  const now = new Date()

  return {
    _id: sessionId,
    name,
    email,
    video1: { promptAttempts: 0 },
    video2: { promptAttempts: 0 },
    createdAt: now,
    updatedAt: now,
  }
}

app.get('/api/health', (_request, response) => {
  response.json({ ok: true })
})

app.post('/api/users/sessions', async (request, response) => {
  const { name, email } = request.body ?? {}

  if (!name || !email) {
    return response.status(400).json({ message: 'Name and email are required.' })
  }

  const sessionId = randomUUID()
  const users = await getUsersCollection()
  const userDocument = buildUserDocument({ sessionId, name, email })

  await users.insertOne(userDocument)

  return response.status(201).json({
    sessionId,
    name,
    email,
    video1: userDocument.video1,
    video2: userDocument.video2,
  })
})

app.post('/api/users/sessions/:sessionId/prompts', async (request, response) => {
  const { sessionId } = request.params
  const { name, email, videoKey } = request.body ?? {}

  if (!name || !email) {
    return response.status(400).json({ message: 'Name and email are required.' })
  }

  if (!['video1', 'video2'].includes(videoKey)) {
    return response.status(400).json({ message: 'Invalid video key.' })
  }

  const users = await getUsersCollection()
  const now = new Date()
  const updateResult = await users.findOneAndUpdate(
    { _id: sessionId },
    {
      $set: {
        name,
        email,
        updatedAt: now,
      },
      $inc: {
        [`${videoKey}.promptAttempts`]: 1,
      },
    },
    { upsert: true, returnDocument: 'after' },
  )

  return response.json({
    sessionId,
    user: updateResult.value,
  })
})

app.post('/api/users/sessions/:sessionId/complete', async (request, response) => {
  const { sessionId } = request.params
  const users = await getUsersCollection()

  await users.updateOne(
    { _id: sessionId },
    {
      $set: {
        completedAt: new Date(),
        updatedAt: new Date(),
      },
    },
  )

  return response.json({ sessionId, completed: true })
})

if (existsSync(clientBuildPath)) {
  app.use(express.static(clientBuildPath))

  app.get('/{*splat}', (_request, response) => {
    response.sendFile(resolve(clientBuildPath, 'index.html'))
  })
}

app.use((error, _request, response, _next) => {
  console.error(error)
  response.status(500).json({ message: 'Internal server error' })
})

app.listen(port, () => {
  console.log(`API server running on http://localhost:${port}`)
})

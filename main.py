# FastAPI application for a simple Todo API (trial)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os

app = FastAPI(title="Todo API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class TodoCreate(BaseModel):
    title: str
    completed: bool = False

class Todo(BaseModel):
    id: int
    title: str
    completed: bool

# In-memory storage (use database in production)
todos: List[Todo] = []
next_id = 1

@app.get("/api/todos", response_model=List[Todo])
async def get_todos():
    return todos

@app.post("/api/todos", response_model=Todo)
async def create_todo(todo: TodoCreate):
    global next_id
    new_todo = Todo(id=next_id, title=todo.title, completed=todo.completed)
    todos.append(new_todo)
    next_id += 1
    return new_todo

@app.put("/api/todos/{todo_id}", response_model=Todo)
async def update_todo(todo_id: int, todo_update: TodoCreate):
    for todo in todos:
        if todo.id == todo_id:
            todo.title = todo_update.title
            todo.completed = todo_update.completed
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")

@app.delete("/api/todos/{todo_id}")
async def delete_todo(todo_id: int):
    global todos
    todos = [todo for todo in todos if todo.id != todo_id]
    return {"message": "Todo deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
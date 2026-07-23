"""⑦ 登入 — OAuth2 password → JWT bearer。

- POST /auth/register  {username,password}          建帳號 → 回 token
- POST /auth/token     form: grant_type=password&username=&password=   → 回 token(OAuth2 標準)
- GET  /auth/me        Authorization: Bearer <jwt>   → 目前使用者

多租戶身分由 token 解出的 user_id 決定;各端點用 get_current_user 依賴取得 user_id。
開發期(AUTH_REQUIRED=false):端點不強制 token(沒帶退回 X-User-Id/DEFAULT_USER),
且 /auth/token 遇未知帳號自動註冊,方便 app「略過登入」與漸進導入。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app import config, db

router = APIRouter(tags=["auth"])


# --- 密碼雜湊 / JWT ---
def _hash(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def _verify(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


def make_jwt(user_id: str, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "username": username,
               "iat": now, "exp": now + timedelta(seconds=config.AUTH_TTL)}
    return jwt.encode(payload, config.AUTH_SECRET, algorithm=config.AUTH_ALGO)


def decode_jwt(token: str) -> str | None:
    """回傳 user_id;無效/過期則 None。"""
    try:
        payload = jwt.decode(token, config.AUTH_SECRET, algorithms=[config.AUTH_ALGO])
        return payload.get("sub")
    except Exception:
        return None


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return None


def _token_response(user: dict) -> dict:
    return {
        "access_token": make_jwt(user["id"], user["username"]),
        "token_type": "bearer",
        "expires_in": config.AUTH_TTL,
        "user_id": user["id"],
        "username": user["username"],
    }


# --- 依賴:從請求解出 user_id ---
async def get_current_user(authorization: str | None = Header(default=None),
                           x_user_id: str | None = Header(default=None)) -> str:
    token = _bearer(authorization)
    if token:
        uid = decode_jwt(token)
        if uid:
            return uid
        if config.AUTH_REQUIRED:
            raise HTTPException(401, "無效或過期的 token")
    if config.AUTH_REQUIRED:
        raise HTTPException(401, "需要登入(Authorization: Bearer)")
    return x_user_id or config.DEFAULT_USER   # 開發期退回身分


# --- 端點 ---
class RegisterReq(BaseModel):
    username: str
    password: str


@router.post("/auth/register")
async def register(req: RegisterReq):
    if not req.username or not req.password:
        raise HTTPException(400, "username / password 不可為空")
    if await db.get_user_by_username(req.username):
        raise HTTPException(409, "帳號已存在")
    user = await db.create_user(db.new_user_id(), req.username, _hash(req.password))
    return _token_response(user)


@router.post("/auth/token")
async def token(form: OAuth2PasswordRequestForm = Depends()):
    user = await db.get_user_by_username(form.username)
    if user is None:
        if config.AUTH_REQUIRED:
            raise HTTPException(401, "帳號或密碼錯誤")
        # 開發期:未知帳號自動註冊
        user = await db.create_user(db.new_user_id(), form.username, _hash(form.password))
    elif not _verify(form.password, user["password_hash"]):
        raise HTTPException(401, "帳號或密碼錯誤")
    return _token_response(user)


@router.get("/auth/me")
async def me(user_id: str = Depends(get_current_user)):
    u = await db.get_user_by_id(user_id)
    return u or {"id": user_id, "username": None}

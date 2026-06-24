export default async () => ({
  "chat.headers": (_input, output) => {
    const headers = output.headers || output
    headers["Modal-Key"] = process.env.MODAL_PROXY_AUTH_TOKEN_ID
    headers["Modal-Secret"] = process.env.MODAL_PROXY_AUTH_TOKEN_SECRET
    headers["Modal-Session-ID"] = process.env.MODAL_SESSION_ID
  },
})

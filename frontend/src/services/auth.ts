export const login = async (email: string, password: string) => {
  return new Promise<{ token: string }>((resolve, reject) => {
    setTimeout(() => {
      if (email === "test@example.com" && password === "password") {
        resolve({ token: "fake-jwt-token" });
      } else {
        reject("Invalid credentials");
      }
    }, 800);
  });
};
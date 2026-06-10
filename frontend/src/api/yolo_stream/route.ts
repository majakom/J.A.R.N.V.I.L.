export const useYoloStream = () => {
  const base = import.meta.env.VITE_API_URL || "http://localhost:8000";

  return `${base}/api/camera/stream/yolo`;
};
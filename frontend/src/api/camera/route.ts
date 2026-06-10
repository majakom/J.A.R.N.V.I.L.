import { api } from "../client/client";
import type { Camera } from "../../types/camera";


export const getCamera = () =>
    api<Camera[]>("/api/camera/status");
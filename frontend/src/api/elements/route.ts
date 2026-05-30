import { api } from "../client/client";
import type {
  Element,
  CreateElementDto,
  UpdateElementDto,
} from "../../types/element";

// GET ALL
export const getElements = () =>
  api<Element[]>("/api/elements/");

// GET ONE
export const getElement = (id: number) =>
  api<Element>(`/api/elements/${id}`);

// CREATE
export const createElement = (data: CreateElementDto) =>
  api<Element>("/api/elements/", {
    method: "POST",
    body: JSON.stringify(data),
  });

// UPDATE
export const updateElement = (id: number, data: UpdateElementDto) =>
  api<Element>(`/api/elements/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });

// DELETE
export const deleteElement = (id: number) =>
  api<void>(`/api/elements/${id}`, {
    method: "DELETE",
  });

//SHOW ELEMENT ON THE TABLE
export const setElementShow = (id: number, show: boolean) =>
    api<Element>(`/api/elements/${id}/show`, {
        method: "POST",
        body: JSON.stringify({ show }),
    });
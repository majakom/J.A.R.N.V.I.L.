export interface Element {
  id: number;
  name: string;
  amount: number;
  url: string;
  comment: string;
}

export interface CreateElementDto {
  name: string;
  amount: number;
  url: string;
  comment: string;
}

export interface UpdateElementDto {
  name?: string;
  amount?: number;
  url?: string;
  comment?: string;
}
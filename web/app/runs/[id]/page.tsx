"use client";

import { useParams } from "next/navigation";

import { RunView } from "@/components/RunView";

export default function RunPage() {
  const { id } = useParams<{ id: string }>();
  return <RunView id={id} />;
}

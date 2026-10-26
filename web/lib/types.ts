// Short names for the API's types. lib/api-schema.d.ts is GENERATED from the backend's OpenAPI schema
// (`npm run types`: python -m shortsmith.api --openapi, then openapi-typescript); never edit it by hand.
// If a Pydantic model in the backend changes and nobody regenerates, tests/test_api.py fails; once it is
// regenerated, every page that used a removed or renamed field fails `npm run build` instead of the browser.
import type { components, paths } from "./api-schema";

type Schemas = components["schemas"];

export type Accepted = Schemas["Accepted"];
export type CostReport = Schemas["CostReport"];
export type Health = Schemas["Health"];
export type Options = Schemas["Options"];
export type Problem = Schemas["Problem"];
export type QueueView = Schemas["QueueView"];
export type RunDetail = Schemas["RunDetail"];
export type RunStatus = Schemas["RunStatus"];
export type RunSummary = Schemas["RunSummary"];
export type StartRun = Schemas["StartRun"];

// Every event the stream sends: the union of GET /runs/{run_id}/log's items (events.py's Event).
export type RunEvent = paths["/runs/{run_id}/log"]["get"]["responses"][200]["content"]["application/json"][number];
export type EventType = RunEvent["type"];
export type EventOf<T extends EventType> = Extract<RunEvent, { type: T }>;

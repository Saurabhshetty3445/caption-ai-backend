-- Run this in your Supabase SQL Editor
-- Dashboard → SQL Editor → New query → paste & run

create table if not exists transcriptions (
  id           uuid primary key,
  filename     text not null,
  transcript   text,
  srt_captions text,
  language     text,
  duration_seconds float,
  status       text default 'pending',
  created_at   timestamptz default now()
);

-- Enable Row Level Security (optional but recommended for SaaS)
alter table transcriptions enable row level security;

-- Allow service role full access (used by backend)
create policy "Service role full access"
  on transcriptions
  for all
  using (true)
  with check (true);

-- Index for fast lookups by created_at
create index if not exists transcriptions_created_at_idx
  on transcriptions (created_at desc);

--CREATE TABLE users (
--    id SERIAL PRIMARY KEY,
--    username VARCHAR(100) UNIQUE NOT NULL,
--    user_role VARCHAR(50) DEFAULT 'user' NOT NULL,
--    is_active BOOLEAN DEFAULT TRUE,
--    is_superuser BOOLEAN DEFAULT FALSE
--);

--CREATE TABLE face_embeddings (
--    id SERIAL PRIMARY KEY,
--    embedding BYTEA NOT NULL,
--    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE
--);


--CREATE TABLE users (
--    id SERIAL PRIMARY KEY,
--    username VARCHAR(100) UNIQUE NOT NULL,
--    user_role VARCHAR(50) DEFAULT 'user' NOT NULL,
--    is_active BOOLEAN DEFAULT TRUE,
--    is_superuser BOOLEAN DEFAULT FALSE,
--    embedding BYTEA NOT NULL
--);

-- Create users table
-- CREATE TABLE users (
--     id SERIAL PRIMARY KEY,
--     username VARCHAR(100) UNIQUE NOT NULL,
--     user_role VARCHAR(50) DEFAULT 'user' NOT NULL,
--     is_active BOOLEAN DEFAULT TRUE,
--     is_superuser BOOLEAN DEFAULT FALSE,
--     embedding BYTEA NOT NULL,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- -- Create evaluation_metrics table
-- CREATE TABLE evaluation_metrics (
--     id SERIAL PRIMARY KEY,
--     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     y_true INTEGER NOT NULL,
--     y_pred INTEGER NOT NULL,
--     accuracy FLOAT,
--     precision FLOAT,
--     recall FLOAT,
--     f1_score FLOAT
-- );




CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    embedding BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE user_admin(
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(20) UNIQUE NOT NULL,
    h_password  VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );


insert into user_admin(username, h_password) values('user@example.com', '$5$rounds=535000$password$ZV/rxEfiphZ3.GBonzeYp/254rV9IMcm0KxcfVtxko7');
--password = sting
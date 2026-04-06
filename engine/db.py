import sqlite3
import os
DB_PATH=os.path.join(os.path.dirname(__file__),'..','data','fingerprints.db')
def get_connection():
    conn=sqlite3.connect(DB_PATH)
    return conn

def store_song(name,hashes):
    conn=get_connection()
    cursor=conn.cursor()
    try:
        cursor.execute("Insert INTO songs (name) VALUES (?)",(name,))
        song_id=cursor.lastrowid
        
        db_hashes=[]
        for h,t in hashes:
            db_hashes.append((h,t,song_id))
            
        cursor.executemany(
            "INSERT INTO fingerprints (hash, time_offset, song_id) VALUES (?, ?, ?)", 
            db_hashes
        )
        conn.commit()
        print(f"Successfully saved'{name}'with {len(hashes)} fingerprints !")
    except sqlite3.IntegrityError:
        print(f"Error: Song '{name}' already exists in the database.")
    finally:
        conn.close()

def setup_db():
    print("Setting up SYNAESTHESIA Database...")
    conn=get_connection()
    cursor=conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fingerprints (
        hash TEXT,
        time_offset INTEGER,
        song_id INTEGER,
        FOREIGN KEY(song_id) REFERENCES songs(id)
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)")
    conn.commit()
    conn.close()
    print("Database is ready at data/fingerprints.db!")

def recognize_audio(target_hashes):
    """
    Takes an array of hashes from a microphone recording, cross-references 
    the database, and figures out what song it is based on time-alignment.
    """
    conn=get_connection()
    cursor=conn.cursor()
    # Extract just the hex string from the target_hashes 
    # (ignoring the time they occurred for the database lookup)
    hash_strings=[h[0] for h in target_hashes]
    placeholders = ','.join(['?'] * len(hash_strings))
    query = f"""
        SELECT hash, time_offset, song_id 
        FROM fingerprints 
        WHERE hash IN ({placeholders})
    """
    cursor.execute(query,hash_strings)
    results=cursor.fetchall()
    matches_per_song={}
    mic_time_dict={}
    for h in target_hashes:
        if h[0] not in mic_time_dict:
            mic_time_dict[h[0]] = []
        mic_time_dict[h[0]].append(h[1])
    
    for db_hash,db_time_offset,song_id in results:
        mic_times=mic_time_dict.get(db_hash, [])
        for mic_time_offset in mic_times:
            time_diff=db_time_offset-mic_time_offset
            if song_id not in matches_per_song:
                matches_per_song[song_id]={}
            if time_diff not in matches_per_song[song_id]:
                matches_per_song[song_id][time_diff]=0
            matches_per_song[song_id][time_diff]+=1
        
    conn.close()
    
    best_song_id=None
    best_match_count=0
    for song_id,time_diffs in matches_per_song.items():
        max_aligned_hashes=max(time_diffs.values()) if time_diffs else 0
        if max_aligned_hashes>10 and max_aligned_hashes>best_match_count:
            best_match_count=max_aligned_hashes
            best_song_id=song_id
            
    if best_song_id:
        conn=get_connection()
        c=conn.cursor()
        c.execute("SELECT name FROM songs WHERE id =?", (best_song_id,))
        song_name=c.fetchone()[0]
        conn.close()
        return f"Match Found: {song_name} ({best_match_count} points of alignment!)"
        
    return "No match found in the database."
    
    



if __name__=="__main__":
    setup_db()



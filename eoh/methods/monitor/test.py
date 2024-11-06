

def get_celebrity_age(name: str) -> int:
    """
    Returns the age of a celebrity based on their name.
    
    Args:
        name (str): The name of the celebrity (case-insensitive)
        
    Returns:
        int: The age of the celebrity
        
    Raises:
        KeyError: If the celebrity is not found in the database
    """
    # Sample celebrity database (you could expand this or connect to an API)
    celebrity_database = {
        "tom cruise": {"birth_year": 1962},
        "jennifer aniston": {"birth_year": 1969},
        "leonardo dicaprio": {"birth_year": 1974},
        "meryl streep": {"birth_year": 1949},
        "morgan freeman": {"birth_year": 1937},
    }
    
    # Get current year
    from datetime import datetime
    current_year = datetime.now().year
    
    # Normalize the input name
    name = name.lower().strip()
    
    # Check if celebrity exists in database
    if name not in celebrity_database:
        raise KeyError(f"Celebrity '{name}' not found in database")
    
    # Calculate age
    birth_year = celebrity_database[name]["birth_year"]
    age = current_year - birth_year
    
    return age




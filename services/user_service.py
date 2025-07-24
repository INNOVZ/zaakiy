from services.supabase_client import client

# Get user with their associated org details


async def get_user_with_org(user_id: str):
    user_response = await client.get("/users", params={"user_id": f"eq.{user_id}", "select": "*,organization(*)"})
    user_data = user_response.json()
    if not user_data:
        raise ValueError("User not found")

    return user_data[0]

# Create a user in the users table if they don't exist


async def sync_user_if_missing(user_id: str, email: str):
    check = await client.get("/users", params={"id": f"eq.{user_id}"})

    if check.status_code == 200 and check.json():
        return check.json()[0]  # User already exists

     # Create a new organization for the user if not found
    org_id = await create_organization_for_user(user_id, email)

    response = await client.post("/users", json={
        "id": user_id,
        "email": email,
        "full_name": "user",
        "org_id": org_id
    })
    if response.status_code >= 400:
        print("Failed to insert user:", response.status_code, response.text)
# ...existing code...


# Create a new organization for the user
async def create_organization_for_user(user_id: str, email: str):
    org_data = {
        "name": f"Organization for {user_id}",
        "email": email
    }

    org_response = await client.post("/organizations", json=org_data)

    print("Org insert status:", org_response.status_code)
    print("Org insert response:", org_response.text)

    if org_response.status_code == 201:
        org_json = org_response.json()
        return org_json["id"]
    else:
        print("Failed to create organization:",
              org_response.status_code, org_response.text)
        raise RuntimeError("Failed to create a new organization")
# ...existing code...

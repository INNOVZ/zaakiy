from services.supabase_client import client

# Get user with their associated org details


async def get_user_with_org(user_id: str):
    # First get the user
    user_response = await client.get("/users", params={"id": f"eq.{user_id}"})
    user_data = user_response.json()
    if not user_data:
        raise ValueError("User not found")

    user = user_data[0]

    # Then get the organization if org_id exists
    if user.get("org_id"):
        org_response = await client.get("/organizations", params={"id": f"eq.{user['org_id']}"})
        org_data = org_response.json()
        if org_data:
            user["organization"] = org_data[0]
        else:
            user["organization"] = None
    else:
        user["organization"] = None

    return user

# ...existing code...

# Create a user in the users table if they don't exist


async def sync_user_if_missing(user_id: str, email: str):
    # Changed from "user_id" to "id"
    check = await client.get("/users", params={"id": f"eq.{user_id}"})

    if check.status_code == 200 and check.json():
        return check.json()[0]  # User already exists

    # Create a new organization for the user if not found
    org_id = await create_organization_for_user(user_id, email)

    user_data = {
        "id": user_id,  # This is correct
        "email": email,
        "full_name": "user",
        "org_id": org_id
    }

    print(f"Creating user with data: {user_data}")  # Debug log
    response = await client.post("/users", json=user_data)

    print(f"User creation status: {response.status_code}")  # Debug log
    print(f"User creation response: {response.text}")  # Debug log

    if response.status_code >= 400:
        print("Failed to insert user:", response.status_code, response.text)
        raise RuntimeError(
            f"Failed to create user: {response.status_code} - {response.text}")

    # Handle empty response body for successful creation
    if response.text.strip():
        return response.json()
    else:
        # Return the user data we just created since Supabase didn't return it
        return user_data
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
    elif org_response.status_code == 409:
        # Organization with this email already exists, fetch it
        existing_org_resp = await client.get("/organizations", params={"email": f"eq.{email}"})
        if existing_org_resp.status_code == 200 and existing_org_resp.json():
            return existing_org_resp.json()[0]["id"]
        else:
            print("Failed to fetch existing organization:",
                  existing_org_resp.status_code, existing_org_resp.text)
            raise RuntimeError("Failed to fetch existing organization")
    else:
        print("Failed to create organization:",
              org_response.status_code, org_response.text)
        raise RuntimeError("Failed to create a new organization")

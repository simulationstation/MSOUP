# CLAUDE.md - Rules for this project

## CRITICAL RULES - NEVER VIOLATE

1. **NEVER DELETE OUTPUT/DATA DIRECTORIES** - Do not use `rm -rf` on any results, output, or data directories. If you need to restart a pipeline, just restart it - do not delete previous work.

2. **DO NOT make "optimizations" that change scientific accuracy** - Only change parallelism/performance settings, never alter algorithms or numerical methods.

3. **When restarting pipelines** - Use environment variables or edit config, do NOT delete output files. Pipelines may have checkpoint/resume capability.

4. **Check Python indentation before editing** - Always check the existing indentation style (spaces vs tabs, indent width) in a Python file BEFORE making edits. Do not assume any paradigm.

5. **No vague time predictions** - Do not say "should finish soon", "wait X minutes", or make vague guesses. Time estimates based on actual evidence (e.g., "batch 1 took 30 min, 9 remaining = ~270 min") are fine.

6. **Never reduce scientific validity or fidelity** - Do not make any edits that could compromise the scientific accuracy of results. Act as an executor, do not infer or make assumptions about scientific requirements.

7. **Check Python tabbing paradigm BEFORE editing** - Before editing any Python file, examine the existing code to determine whether it uses tabs or spaces (and how many spaces). Match the existing style exactly.

## Globus Data Transfer Setup

To download SDSS data via Globus:

1. **Extract Globus Connect Personal and config:**
   ```bash
   tar -xzf /home/primary/globusconnectpersonal-latest.tgz
   tar -xzf /home/primary/globusonline.tar.gz -C ~/
   ```

2. **Start GCP daemon:**
   ```bash
   cd ~/globusconnectpersonal-*/
   ./globusconnectpersonal -start &
   ./globusconnectpersonal -status  # Should show "connected"
   ```

3. **Authenticate and transfer using Python SDK:**
   ```python
   from globus_sdk import NativeAppAuthClient, TransferClient, AccessTokenAuthorizer, TransferData
   from globus_sdk.scopes import TransferScopes

   # Use GCP's client ID
   CLIENT_ID = "4d6448ae-8ca0-40e4-aaa9-8ec8e8320621"
   client = NativeAppAuthClient(CLIENT_ID)
   client.oauth2_start_flow(requested_scopes=TransferScopes.all, refresh_tokens=True)

   # Get auth code from URL, exchange for tokens
   url = client.oauth2_get_authorize_url()
   # User visits URL, gets code
   tokens = client.oauth2_exchange_code_for_tokens(auth_code)

   # Create transfer client and submit transfer
   tc = TransferClient(authorizer=AccessTokenAuthorizer(tokens.by_resource_server["transfer.api.globus.org"]["access_token"]))

   SDSS_EP = "f8362eaf-fc40-451c-8c44-50b71ec7f247"  # SDSS Public Data Release
   LOCAL_EP = "d1294d08-e055-11f0-a4db-0213754b0ca1"  # Our GCP endpoint

   tdata = TransferData(source_endpoint=SDSS_EP, destination_endpoint=LOCAL_EP, label="SDSS download")
   tdata.add_item("/dr16/eboss/lss/catalogs/DR16/FILE.fits", "/local/path/FILE.fits")
   result = tc.submit_transfer(tdata)
   ```

4. **Key endpoints:**
   - SDSS Public Data: `f8362eaf-fc40-451c-8c44-50b71ec7f247`
   - Local GCP (MSOUP-Analysis): `d1294d08-e055-11f0-a4db-0213754b0ca1`

5. **Tokens saved to:** `~/.globus_tokens.json`

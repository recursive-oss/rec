package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func downloadClient() *http.Client {
	return &http.Client{Transport: &http.Transport{Proxy: http.ProxyFromEnvironment, ResponseHeaderTimeout: 30 * time.Second}, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 {
			return fmt.Errorf("too many redirects")
		}
		if req.URL.Scheme != "https" {
			return fmt.Errorf("refusing non-HTTPS redirect")
		}
		if req.URL.Host != via[0].URL.Host {
			req.Header.Del("Authorization")
		}
		return nil
	}}
}

func download(ctx context.Context, client *http.Client, f File, dest string, progress *transfer, base int64) error {
	part := dest + ".partial"
	var offset int64
	if st, e := os.Lstat(part); e == nil {
		if !st.Mode().IsRegular() {
			return fmt.Errorf("invalid partial file")
		}
		offset = st.Size()
	}
	if offset > f.Size {
		if err := os.Remove(part); err != nil {
			return err
		}
		offset = 0
	}
	if offset == f.Size {
		progress.set(base+offset, "verifying "+f.Name)
		if err := verifyFile(part, f); err == nil {
			return os.Rename(part, dest)
		}
		os.Remove(part)
		offset = 0
	}
	requestCtx, cancel := context.WithTimeout(ctx, 2*time.Hour)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, "GET", f.URL, nil)
	if err != nil {
		return err
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("download %s: HTTP %d", f.Name, resp.StatusCode)
	}
	if resp.StatusCode == http.StatusPartialContent {
		cr := resp.Header.Get("Content-Range")
		var start, end, total int64
		n, e := fmt.Sscanf(cr, "bytes %d-%d/%d", &start, &end, &total)
		if e != nil || n != 3 || start != offset || total != f.Size || end != f.Size-1 {
			return fmt.Errorf("invalid Content-Range: %s", cr)
		}
	} else {
		offset = 0
	}
	remaining := f.Size - offset
	if resp.ContentLength >= 0 && resp.ContentLength != remaining {
		return fmt.Errorf("unexpected download length for %s", f.Name)
	}
	flags := os.O_CREATE | os.O_RDWR
	if offset == 0 {
		flags |= os.O_TRUNC
	}
	file, err := os.OpenFile(part, flags, 0600)
	if err != nil {
		return err
	}
	progress.set(base+offset, "")
	hash := sha256.New()
	_, copyErr := io.CopyN(hash, file, offset)
	var n int64
	if copyErr == nil {
		n, copyErr = io.Copy(io.MultiWriter(file, hash, progress), io.LimitReader(resp.Body, remaining+1))
	}
	syncErr := file.Sync()
	closeErr := file.Close()
	if copyErr == nil && n != remaining {
		copyErr = fmt.Errorf("incomplete download %s: received %s of %s bytes; retry to resume", f.Name, strconv.FormatInt(n, 10), strconv.FormatInt(remaining, 10))
	}
	if copyErr == nil {
		copyErr = syncErr
	}
	if copyErr == nil {
		copyErr = closeErr
	}
	if copyErr != nil {
		return copyErr
	}
	if hex.EncodeToString(hash.Sum(nil)) != f.SHA256 {
		os.Remove(part)
		return fmt.Errorf("SHA-256 mismatch: %s", f.Name)
	}
	return os.Rename(part, dest)
}

// This is only a launcher scheduling hint, never an integrity check.
func modelFilesPresent(p Paths, m Manifest) bool {
	root := filepath.Join(p.Data, "models", modelDigest(m))
	for _, f := range m.Files {
		if !cachedSize(filepath.Join(root, f.Name), f) {
			return false
		}
	}
	return true
}

func ensureModel(ctx context.Context, p Paths, m Manifest, out io.Writer) (string, error) {
	client := downloadClient()
	defer client.CloseIdleConnections()
	return ensureModelWithClient(ctx, p, m, out, client)
}

// verifyLocalModel hashes the manifest-declared files in their existing
// directory. It never copies or downloads model bytes; the native runtime
// independently revalidates the pinned checkpoint before GPU admission.
func verifyLocalModel(root string, m Manifest, out io.Writer) (string, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	before, err := os.Lstat(abs)
	if err != nil || !before.IsDir() || before.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("local model directory must be a real directory")
	}
	var total int64
	for _, f := range m.Files {
		total += f.Size
	}
	progress := newTransfer(out, "Verifying "+displayName(m.Model), "Verified "+displayName(m.Model), 0, total)
	for _, f := range m.Files {
		if err := verifyFileInto(filepath.Join(abs, f.Name), f, progress); err != nil {
			progress.fail(err.Error())
			return "", err
		}
	}
	after, err := os.Lstat(abs)
	if err != nil || !os.SameFile(before, after) {
		return "", fmt.Errorf("local model directory changed during verification")
	}
	progress.succeed()
	return filepath.Join(abs, m.Entry), nil
}

func ensureModelWithClient(ctx context.Context, p Paths, m Manifest, out io.Writer, client *http.Client) (string, error) {
	id := modelDigest(m)
	root := filepath.Join(p.Data, "models", id)
	if err := os.MkdirAll(root, 0700); err != nil {
		return "", err
	}
	l, err := lock(filepath.Join(root, "download.lock"))
	if err != nil {
		return "", err
	}
	defer unlock(l)
	verified, err := loadVerified(root)
	if err != nil {
		return "", err
	}
	trusted := map[string]bool{}
	var unverified int64
	for _, f := range m.Files {
		trusted[f.Name] = verified.trusted(f)
		if !trusted[f.Name] && cachedSize(filepath.Join(root, f.Name), f) {
			unverified += f.Size
		}
	}
	name := displayName(m.Model)
	var verifying *transfer
	var verifyErr error
	if unverified > 0 {
		verifying = newTransfer(out, "Verifying "+name, "Verified "+name, 0, unverified)
	}
	var need, total int64
	missing := []File{}
	for _, f := range m.Files {
		path := filepath.Join(root, f.Name)
		if trusted[f.Name] {
			continue
		}
		if verifying != nil && cachedSize(path, f) {
			err := verified.verify(f, verifying)
			if err == nil {
				continue
			}
			verifyErr = err
		}
		missing = append(missing, f)
		remaining := f.Size
		if st, err := os.Lstat(path + ".partial"); err == nil && st.Mode().IsRegular() && st.Size() < f.Size {
			remaining -= st.Size()
		}
		need += remaining
		total += f.Size
	}
	if verifyErr != nil {
		verifying.fail(verifyErr.Error() + "; downloading again")
	} else if verifying != nil {
		verifying.succeed()
	}
	if need > 0 {
		if err := diskSpace(root, need+64*1024*1024); err != nil {
			return "", err
		}
	}
	if len(missing) == 0 {
		return filepath.Join(root, m.Entry), nil
	}
	progress := newTransfer(out, "Downloading "+name, "Downloaded "+name, total-need, total)
	var base int64
	for _, f := range missing {
		dest := filepath.Join(root, f.Name)
		var err error
		for attempt := 0; attempt < 3; attempt++ {
			err = download(ctx, client, f, dest, progress, base)
			if err == nil {
				break
			}
			if ctx.Err() != nil {
				progress.fail("")
				return "", ctx.Err()
			}
			if strings.Contains(err.Error(), "SHA-256") {
				break
			}
			if attempt < 2 {
				progress.note(fmt.Sprintf("Retrying %s: %v", f.Name, err))
				select {
				case <-ctx.Done():
					progress.fail("")
					return "", ctx.Err()
				case <-time.After(time.Duration(attempt+1) * time.Second):
				}
			}
		}
		if err == nil {
			err = verified.record(f)
		}
		if err != nil {
			progress.fail("")
			return "", err
		}
		base += f.Size
	}
	progress.succeed()
	return filepath.Join(root, m.Entry), nil
}

func verifyModel(p Paths, m Manifest, out io.Writer) (string, error) {
	id := modelDigest(m)
	root := filepath.Join(p.Data, "models", id)
	l, err := lock(filepath.Join(root, "download.lock"))
	if err != nil {
		return "", err
	}
	defer unlock(l)
	verified, err := loadVerified(root)
	if err != nil {
		return "", err
	}
	var total int64
	for _, f := range m.Files {
		total += f.Size
	}
	name := displayName(m.Model)
	verifying := newTransfer(out, "Verifying "+name, "Verified "+name, 0, total)
	for _, f := range m.Files {
		if err := verified.verify(f, verifying); err != nil {
			verifying.fail(err.Error())
			return "", errors.Join(err, verified.forget(f))
		}
	}
	verifying.succeed()
	return id, nil
}
